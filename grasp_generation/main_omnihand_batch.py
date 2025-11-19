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
parser.add_argument('--w_spen', default=10.0, type=float)
parser.add_argument('--w_joints', default=1.0, type=float)
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
grasp_file = "dexycb_robot_joint_dict_1030_1525_omnihand"
result_path = "/home/guizhewei/guizhewei/grasp_pose_dataset/unoptimized"
data_dict = np.load(os.path.join(result_path, grasp_file + '.npy'), allow_pickle=True)
object_code_list = []
hand_pose_list = []
obj_idx = []
for data in data_dict:
    if args.object_code is not None and data['object_code'] != args.object_code:
        continue
    qpos = data['qpos']
    object_code_list.append(data['object_code'])
    obj_idx.append(data['idx'])
    object_pose = data['object_pose']
    rot = data['hand_rot6d']
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
    hand_pose_list.append(torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device))
    # if args.object_code is not None:
    #     break
# 检查是否加载到数据
if len(hand_pose_list) == 0:
    if args.object_code is not None:
        raise ValueError(f"未找到匹配的 object_code: {args.object_code}。请检查数据文件中是否存在该对象代码。")
    else:
        raise ValueError("数据文件为空，未加载到任何数据。")

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
    data_root_path='/home/guizhewei/guizhewei/Dexycb_dataset/models',
    batch_size_each=args.batch_size,
    num_samples=2000,
    device=device
)
object_model.initialize(object_code_list=object_code_list)

hand_st_plotly = []
# contact_point_indices = torch.randint(hand_model.n_contact_candidates, size=[total_batch_size, args.n_contact], device=device)
contact_point_indices = torch.arange(
    hand_model.n_contact_candidates, device=device
).repeat(total_batch_size, 1)
hand_pose_tensor.requires_grad_()
hand_model.set_parameters(hand_pose_tensor, contact_point_indices)

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
logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, 0, show=True)
pbar = tqdm(range(1, args.n_iter + 1), desc='optimizing', dynamic_ncols=True)
for step in pbar:
    s = optimizer.try_step(fix_wrist=args.fix_wrist)

    optimizer.zero_grad()
    new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    new_energy.sum().backward(retain_graph=True)

    with torch.no_grad():
        accept, t = optimizer.accept_step(energy, new_energy)

        energy[accept] = new_energy[accept]
        E_dis[accept] = new_E_dis[accept]
        E_fc[accept] = new_E_fc[accept]
        E_pen[accept] = new_E_pen[accept]
        E_spen[accept] = new_E_spen[accept]
        E_joints[accept] = new_E_joints[accept]

        logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, step, show=False)

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
            qpos=qpos,
            qpos_st=qpos_st,
            energy=energy[idx].item(),
            E_fc=E_fc[idx].item(),
            E_dis=E_dis[idx].item(),
            E_pen=E_pen[idx].item(),
            E_spen=E_spen[idx].item(),
            E_joints=E_joints[idx].item(),
            idx=obj_idx[i],
        ))
    np.save(os.path.join(result_path, str(obj_idx[i]) + '_' + object_code_list[i] + '.npy'), data_list, allow_pickle=True)
    print("Saved to ", result_path)

