"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
"""

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go

from utils.hand_model import HandModel
from utils.object_model import ObjectModel

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code', type=str, default='0_7c65b27279444df5b70bc17e9e1a6c70')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='/home/guizhewei/guizhewei/DexGraspNet/data/experiments/dexonomy_1obj_pen_spen_contact/results')
    parser.add_argument('--object_surface_points', action='store_true', help='可视化物体表面采样点的SDF')
    parser.add_argument('--sdf_point_size', type=float, default=2, help='SDF采样点大小')
    parser.add_argument('--sdf_min', type=float, default=-0.01, help='SDF颜色条最小值')
    parser.add_argument('--sdf_max', type=float, default=0.01, help='SDF颜色条最大值')
    args = parser.parse_args()

    device = 'cpu'
    
    # 如果需要可视化表面点SDF，需要CUDA
    if args.object_surface_points and not torch.cuda.is_available():
        print("警告: --object_surface_points 需要CUDA支持（TorchSDF需要）")
        print("将跳过表面点SDF可视化")
        args.object_surface_points = False

    # load results
    data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[args.num]
    qpos = data_dict['qpos']
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
    if 'qpos_st' in data_dict:
        qpos_st = data_dict['qpos_st']
        rot = np.array(transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot + [qpos_st[name] for name in joint_names], dtype=torch.float, device=device)
        # print("hand_pose_st", hand_pose_st)
        # print("hand_pose_st.shape", hand_pose_st.shape)
        # print("qpos_st", qpos_st)
        # import ipdb; ipdb.set_trace()
    # Temporarily change to mjcf directory for mesh loading
    current_dir = os.getcwd()
    os.chdir('/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf')

    hand_model = HandModel(
        mjcf_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf/omnihand_tendon_contact.xml',
        mesh_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf',
        contact_points_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/contact_points_omnihand.json',
        penetration_points_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/penetration_points_omnihand.json',
        device=device
        )

    # Change back to original directory
    os.chdir(current_dir)

    object_model = ObjectModel(
        data_root_path='/home/guizhewei/guizhewei/DexGraspNet/data/1_obj',
        batch_size_each=1,
        num_samples=2000,
        device=device
    )
    name_no_ext = os.path.splitext(args.object_code)[0]
    object_idx, object_code = name_no_ext.split("_", 1)
    
    # 读取 scene_scale 和 obj_scale（如果存在）
    scene_scale = None
    obj_scale = None
    if 'scene_scale' in data_dict:
        scene_scale = data_dict['scene_scale']
    if 'obj_scale' in data_dict:
        obj_scale = data_dict['obj_scale']
    
    # 使用读取的 scale 初始化 object_model
    object_model.initialize(object_code_list=object_code, scene_scale_list=[scene_scale], obj_scale_list=[obj_scale])

    # visualize

    if 'qpos_st' in data_dict:
        hand_model.set_parameters(hand_pose_st.unsqueeze(0))
        hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightgray', with_contact_points=False)
    else:
        hand_st_plotly = []
    hand_model.set_parameters(hand_pose.unsqueeze(0))
    
    # info
    surface_points = hand_model.get_surface_points()[0].detach().cpu().numpy()
    contact_candidates = hand_model.get_contact_candidates()[0].detach().cpu().numpy()
    penetration_keypoints = hand_model.get_penetraion_keypoints()[0].detach().cpu().numpy()
    
    print('n_surface_points', surface_points.shape[0])
    print('n_contact_candidates', contact_candidates.shape[0])
    
    
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=0.4)
    
    # 可选: 可视化物体表面采样点的SDF
    surface_points_plotly = []
    if args.object_surface_points:
        try:
            # 获取物体表面采样点（与energy.py中E_pen计算使用的相同）
            object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
            object_surface_points_scaled = object_model.surface_points_tensor * object_scale
            
            # 将数据转移到CUDA进行计算
            if device == 'cpu':
                print("\n注意: 为计算SDF，临时将数据转移到CUDA...")
                device_temp = 'cuda'
                object_surface_points_cuda = object_surface_points_scaled.to(device_temp)
                hand_model_temp = hand_model
                # 临时将hand_model的数据转到CUDA
                hand_pose_cuda = hand_pose.to(device_temp)
                
                # 重新创建hand_model在CUDA上
                current_dir_temp = os.getcwd()
                os.chdir('/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf')
                hand_model_cuda = HandModel(
                    mjcf_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf/omnihand_tendon_contact.xml',
                    mesh_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf',
                    contact_points_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/contact_points_omnihand.json',
                    penetration_points_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/penetration_points_omnihand.json',
                    device=device_temp
                )
                os.chdir(current_dir_temp)
                hand_model_cuda.set_parameters(hand_pose_cuda.unsqueeze(0))
                
                # 计算距离
                distances = hand_model_cuda.cal_distance(object_surface_points_cuda)
            else:
                distances = hand_model.cal_distance(object_surface_points_scaled)
            
            # 转换到CPU用于可视化
            object_surface_points = object_surface_points_scaled[0].detach().cpu().numpy()
            distances_np = distances[0].detach().cpu().numpy()
            
            # 打印穿透统计
            n_penetrating = (distances_np > 0).sum()
            n_outside = (distances_np <= 0).sum()
            max_penetration = distances_np.max()
            min_distance = distances_np.min()
            print(f'\n{"="*50}')
            print(f'物体表面采样点SDF统计')
            print(f'{"="*50}')
            print(f'总采样点数:        {len(distances_np)}')
            print(f'穿透点数 (SDF>0):  {n_penetrating} ({100*n_penetrating/len(distances_np):.2f}%)')
            print(f'外部点数 (SDF<=0): {n_outside} ({100*n_outside/len(distances_np):.2f}%)')
            print(f'最大穿透深度:      {max_penetration:.6f}')
            print(f'最小距离值:        {min_distance:.6f}')
            print(f'{"="*50}\n')
            
            # 创建表面点可视化
            surface_points_plotly = [go.Scatter3d(
                x=object_surface_points[:, 0],
                y=object_surface_points[:, 1],
                z=object_surface_points[:, 2],
                mode='markers',
                marker=dict(
                    size=args.sdf_point_size,
                    color=distances_np,
                    colorscale='RdYlGn_r',  # 红色=穿透, 绿色=外部
                    colorbar=dict(
                        title='SDF值<br>(正=穿透)',
                        thickness=15,
                        len=0.6,
                        x=1.1  # 放在右侧
                    ),
                    cmin=args.sdf_min,
                    cmax=args.sdf_max,
                    showscale=True,
                    line=dict(width=0)
                ),
                name='物体表面采样点',
                hovertemplate='<b>采样点</b><br>' +
                             '坐标: (%{x:.4f}, %{y:.4f}, %{z:.4f})<br>' +
                             'SDF: %{marker.color:.6f}<br>' +
                             '<extra></extra>'
            )]
        except Exception as e:
            print(f"\n警告: 无法计算表面点SDF: {e}")
            print("将继续显示其他可视化内容")
            import traceback
            traceback.print_exc()
    
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly + surface_points_plotly)
    if 'energy' in data_dict:
        energy = data_dict['energy']
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        # print("E_pen: ", data_dict.get('E_pen', 0))
        E_spen = round(data_dict.get('E_spen', 0), 5)
        E_joints = round(data_dict['E_joints'], 5)
        E_cmap = round(data_dict.get('E_cmap', 0), 5)
        result = f'Index {args.num}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}  E_spen {E_spen}  E_joints {E_joints}  E_cmap {E_cmap}'
        fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
    fig.update_layout(scene_aspectmode='data')
    fig.show()
