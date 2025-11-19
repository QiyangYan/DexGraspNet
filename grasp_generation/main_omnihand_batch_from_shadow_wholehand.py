"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: Entry of the program, generate small-scale experiments for OmniHand
"""

import os

os.chdir(os.path.dirname(__file__))

import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
import transforms3d
import plotly.graph_objects as go

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.logger import Logger
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import plotly.io as pio
# 若本机有 Firefox/Chromium
pio.renderers.default = "firefox" 

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "thumb_mcp",
    "thumb_ip",
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",
    "index_dip",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "middle_dip",
    "ring_mcp_pitch",
    "ring_pip",
    "ring_dip",
    "pinky_mcp_pitch",
    "pinky_pip",
    "pinky_dip"
]

# prepare arguments

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--num', default=0, type=int)
parser.add_argument('--object_code', default=None, type=str)
parser.add_argument('--name', default='exp_2', type=str)
parser.add_argument('--n_contact', default=96, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--n_iter', default=6000, type=int)
parser.add_argument('--fix_wrist', action='store_true', default=False, help='fix wrist translation and rotation')
# hyper parameters (** Magic, don't touch! **)
parser.add_argument('--switch_possibility', default=0.5, type=float)
parser.add_argument('--mu', default=0.98, type=float)
parser.add_argument('--step_size', default=0.005, type=float)
parser.add_argument('--stepsize_period', default=50, type=int)
parser.add_argument('--starting_temperature', default=18, type=float)
parser.add_argument('--annealing_period', default=30, type=int)
parser.add_argument('--temperature_decay', default=0.95, type=float)
parser.add_argument('--w_dis', default=100.0, type=float)
parser.add_argument('--w_pen', default=100.0, type=float)
parser.add_argument('--w_spen', default=100.0, type=float)
parser.add_argument('--w_joints', default=1.0, type=float)
parser.add_argument('--w_cmap', default=100.0, type=float, help='weight for contact map loss')
parser.add_argument('--cmap_energy_func', default='align_dist', type=str, choices=['align_dist', 'euclidean_dist'], 
                    help='contact map energy function')
# initialization settings
parser.add_argument('--jitter_strength', default=0.1, type=float)
parser.add_argument('--distance_lower', default=0.2, type=float)
parser.add_argument('--distance_upper', default=0.3, type=float)
parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
# energy thresholds
parser.add_argument('--thres_fc', default=0.3, type=float)
parser.add_argument('--thres_dis', default=0.005, type=float)
parser.add_argument('--thres_pen', default=0.001, type=float)

args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# prepare models
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)


# TODO： modify the grasp file name
grasp_file = "dexonomy_1obj_wi_contact_map"
result_path = "/home/guizhewei/guizhewei/grasp_pose_dataset/unoptimized"
# 加载所有数据（去掉 [:10] 限制）
data_dict = np.load(os.path.join(result_path, grasp_file + '.npy'), allow_pickle=True)
print(f"成功加载 {len(data_dict)} 个抓取姿态")

object_code_list = []
hand_pose_list = []
obj_idx = []
scene_scale_list = []
obj_scale_list = []
contact_map_list = []  # 存储 contact map 数据
skipped_count = 0

# TODO: debug first 10
data_dict = data_dict[:1]

for i, data in enumerate(data_dict):
    if args.object_code is not None and data['object_code'] != args.object_code:
        continue
    qpos = data['qpos']
    object_code_list.append(data['object_code'])
    obj_idx.append(data['idx'])
    object_pose = data['object_pose']
    rot = data['hand_rot6d']
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
    hand_pose_list.append(torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device))
    
    # 读取 scene_scale 和 obj_scale
    if 'scene_scale' in data:
        scene_scale_list.append(data['scene_scale'])
    else:
        scene_scale_list.append(None)
    
    if 'obj_scale' in data:
        obj_scale_list.append(data['obj_scale'])
    else:
        obj_scale_list.append(None)
    
    # 读取 contact map 相关数据
    contact_map_data = {}
    if 'object_point_cloud' in data and 'object_normal_cloud' in data and 'contact_map_object' in data:
        contact_map_data['object_point_cloud'] = data['object_point_cloud']  # (M, 3)
        contact_map_data['object_normal_cloud'] = data['object_normal_cloud']  # (M, 3)
        contact_map_data['contact_value'] = data['contact_map_object']  # (M,)
        # print(f"  加载 contact map: obj_points={contact_map_data['object_point_cloud'].shape}, "
        #       f"contact_value range=[{contact_map_data['contact_value'].min():.3f}, {contact_map_data['contact_value'].max():.3f}]")
    else:
        print(f"  警告: 数据中缺少 contact map 信息")
    contact_map_list.append(contact_map_data)
    
    if args.object_code is not None:
        break

hand_pose_tensor = torch.stack([hp.to('cuda:0').view(-1) for hp in hand_pose_list], dim=0)

total_batch_size = len(object_code_list) * args.batch_size

# Temporarily change to mjcf directory for mesh loading
current_dir = os.getcwd()
os.chdir('/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf')

hand_model = HandModel(
    mjcf_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf/omnihand_tendon_contact.xml',
    mesh_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf',
    contact_points_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/contact_points_omnihand.json',
    penetration_points_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/penetration_points_omnihand.json',
    n_surface_points = 2000,
    device=device
    )

# Change back to original directory
os.chdir(current_dir)

object_model = ObjectModel(
    data_root_path='/home/guizhewei/guizhewei/DexGraspNet/data/1_obj',
    batch_size_each=args.batch_size,
    num_samples=2000,
    device=device
)
object_model.initialize(object_code_list=object_code_list, scene_scale_list=scene_scale_list, obj_scale_list=obj_scale_list)

# 打印 scale 信息用于调试
print("\n=== Scale 信息 ===")
for i in range(len(object_code_list)):
    print(f"Object {i}: {object_code_list[i]}")
    print(f"  scene_scale: {scene_scale_list[i]}")
    print(f"  obj_scale: {obj_scale_list[i]}")
    if scene_scale_list[i] is not None and obj_scale_list[i] is not None:
        if isinstance(obj_scale_list[i], np.ndarray):
            obj_scale_val = np.mean(obj_scale_list[i]) if obj_scale_list[i].ndim == 1 else obj_scale_list[i].flat[0]
        else:
            obj_scale_val = obj_scale_list[i]
        total_scale = float(scene_scale_list[i]) * float(obj_scale_val)
        print(f"  total_scale (scene * obj): {total_scale}")
    print(f"  object_scale_tensor: {object_model.object_scale_tensor[i][0].item()}")
print("=" * 20 + "\n")

hand_st_plotly = []
# contact_point_indices = torch.randint(hand_model.n_contact_candidates, size=[total_batch_size, args.n_contact], device=device)
contact_point_indices = torch.arange(
    hand_model.n_contact_candidates, device=device
).repeat(total_batch_size, 1)
hand_pose_tensor.requires_grad_()
hand_model.set_parameters(hand_pose_tensor, contact_point_indices)

# 设置 contact map goal（如果数据中包含）
print("\n=== 设置 Contact Map Goal ===")
for i, contact_map_data in enumerate(contact_map_list):
    if len(contact_map_data) > 0:
        # 组合为 (N, 7) 格式: [x, y, z, nx, ny, nz, contact_value]
        obj_pc = contact_map_data['object_point_cloud']  # (N, 3)
        obj_normal = contact_map_data['object_normal_cloud']  # (N, 3)
        contact_val = contact_map_data['contact_value']  # (N,)
        
        # 转换为 torch tensor
        contact_map_goal = np.concatenate([
            obj_pc,
            obj_normal,
            contact_val.reshape(-1, 1)
        ], axis=1)  # (N, 7)
        
        contact_map_goal_tensor = torch.from_numpy(contact_map_goal).float().to(device)
        
        # 设置 contact map goal（所有 batch 共享同一个 object）
        hand_model.set_contact_map_goal(contact_map_goal_tensor)
        goal_fig = hand_model.visualize_cmap(obj_pc, contact_val, show_hand=False)
        goal_fig.show()
        
        print(f"Object {i}: 设置 contact map goal, shape={contact_map_goal_tensor.shape}")
        print(f"  Contact value stats: mean={contact_val.mean():.4f}, "
              f"max={contact_val.max():.4f}, min={contact_val.min():.4f}")
        
        # 只设置第一个 object 的 contact map（因为 hand_model 是单个实例）
        # 如果需要支持多个 object，需要在优化循环中动态设置
        break
    else:
        print(f"Object {i}: 未找到 contact map 数据，跳过")

if all(len(cmap) == 0 for cmap in contact_map_list):
    print("警告: 所有数据都没有 contact map 信息，contact map loss 将为 0")
print("=" * 40 + "\n")

# for i in range(5):
#     hand_en_plotly = hand_model.get_plotly_data(i=i, opacity=1, color='lightblue', with_contact_points=False)
#     object_plotly = object_model.get_plotly_data(i=i, color='lightgreen', opacity=1)
#     fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
#     fig.update_layout(scene_aspectmode='data')
#     fig.show()
# input("Verify the pose for optimization, press Enter to continue...")

# print('n_contact_candidates', hand_model.n_contact_candidates)
# print('total batch size', total_batch_size)
hand_pose_st = hand_model.hand_pose.detach()

optim_config = {
    'switch_possibility': args.switch_possibility,
    'starting_temperature': args.starting_temperature,
    'temperature_decay': args.temperature_decay,
    'annealing_period': args.annealing_period,
    'step_size': args.step_size,
    'stepsize_period': args.stepsize_period,
    'mu': args.mu,
    'device': device
}
optimizer = Annealing(hand_model, init_hand_pose=hand_pose_tensor, **optim_config)

try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'logs'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'logs'), exist_ok=True)
logger_config = {
    'thres_fc': args.thres_fc,
    'thres_dis': args.thres_dis,
    'thres_pen': args.thres_pen
}
logger = Logger(log_dir=os.path.join('../data/experiments', args.name, 'logs'), **logger_config)


# log settings
with open(os.path.join('../data/experiments', args.name, 'output.txt'), 'w') as f:
    f.write(str(args) + '\n')

# optimize
weight_dict = dict(
    w_dis=args.w_dis,
    w_pen=args.w_pen,
    w_spen=args.w_spen,
    w_joints=args.w_joints,
    w_cmap=args.w_cmap,
    cmap_energy_func=args.cmap_energy_func,
)
energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_cmap = cal_energy(hand_model, object_model, verbose=True, **weight_dict)
print('Initial energy:', energy.mean().item(),
      ' E_fc:', E_fc.mean().item(),
      ' E_dis:', E_dis.mean().item(),
      ' E_pen:', E_pen.mean().item(),   
    ' E_spen:', E_spen.mean().item(),
        ' E_joints:', E_joints.mean().item(),
        ' E_cmap:', E_cmap.mean().item())
energy.sum().backward(retain_graph=True)
logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, 0, E_cmap=E_cmap, show=True)
pbar = tqdm(range(1, args.n_iter + 1), desc='optimizing', dynamic_ncols=True)
from termcolor import cprint
for step in pbar:
    s = optimizer.try_step(fix_wrist=args.fix_wrist)

    optimizer.zero_grad()
    new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints, new_E_cmap = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    new_energy.sum().backward(retain_graph=True)

    with torch.no_grad():
        accept, t = optimizer.accept_step(energy, new_energy)
        

        if step % 250 == 0:
            current_fig = hand_model.visualize_cmap(hand_model.object_point_cloud.cpu().numpy(), hand_model.contact_value_current[0].cpu().numpy(), show_hand=True)

            cprint(f"Step: {step} | Cmap Loss: {E_cmap.mean().item():.3f}", 'green')
            current_fig.show()

            from ipdb import set_trace; set_trace()

        energy[accept] = new_energy[accept]
        E_dis[accept] = new_E_dis[accept]
        E_fc[accept] = new_E_fc[accept]
        E_pen[accept] = new_E_pen[accept]
        E_spen[accept] = new_E_spen[accept]
        E_joints[accept] = new_E_joints[accept]
        E_cmap[accept] = new_E_cmap[accept]

        logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, step, E_cmap=E_cmap, show=False)

        pbar.set_postfix({
            "E": f"{energy.mean().item():.3f}",
            "fc": f"{E_fc.mean().item():.3f}",
            "dis": f"{E_dis.mean().item():.3f}",
            "pen": f"{E_pen.mean().item():.3f}",
            "spen": f"{E_spen.mean().item():.3f}",
            "joints": f"{E_joints.mean().item():.3f}",
            "cmap": f"{E_cmap.mean().item():.3f}"
        })


# save results
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "thumb_mcp",
    "thumb_ip",
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",
    "index_dip",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "middle_dip",
    "ring_mcp_pitch",
    "ring_pip",
    "ring_dip",
    "pinky_mcp_pitch",
    "pinky_pip",
    "pinky_dip"
]

try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'results'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'results'), exist_ok=True)
result_path = os.path.join('../data/experiments', args.name, 'results')
os.makedirs(result_path, exist_ok=True)
for i in range(len(object_code_list)):
    data_list = []
    for j in range(args.batch_size):
        idx = i * args.batch_size + j
        scale = object_model.object_scale_tensor[i][j].item()
        scene_scale = object_model.scene_scale_tensor[i][j].item()
        obj_scale = object_model.obj_scale_tensor[i][j].item()
        hand_pose = hand_model.hand_pose[idx].detach().cpu()
        qpos = dict(zip(joint_names, hand_pose[9:].tolist()))
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
        euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        qpos.update(dict(zip(rot_names, euler)))
        qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))
        hand_pose = hand_pose_st[idx].detach().cpu()
        qpos_st = dict(zip(joint_names, hand_pose[9:].tolist()))
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
        euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        qpos_st.update(dict(zip(rot_names, euler)))
        qpos_st.update(dict(zip(translation_names, hand_pose[:3].tolist())))
        data_list.append(dict(
            scale=scale,
            scene_scale=scene_scale,
            obj_scale=obj_scale,
            qpos=qpos,
            qpos_st=qpos_st,
            energy=energy[idx].item(),
            E_fc=E_fc[idx].item(),
            E_dis=E_dis[idx].item(),
            E_pen=E_pen[idx].item(),
            E_spen=E_spen[idx].item(),
            E_joints=E_joints[idx].item(),
            E_cmap=E_cmap[idx].item(),
            idx=obj_idx[i],
        ))
    np.save(os.path.join(result_path, str(obj_idx[i]) + '_' + object_code_list[i] + '.npy'), data_list, allow_pickle=True)
    print("Saved to ", result_path)

