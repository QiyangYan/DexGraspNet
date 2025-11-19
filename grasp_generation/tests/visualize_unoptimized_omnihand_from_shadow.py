"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize unoptimized grasp data from shadow retarget to omnihand using plotly.graph_objects
"""

import os
import sys

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
    parser.add_argument('--grasp_file', type=str, default='dexonomy_1obj_fin_cmap')
    parser.add_argument('--result_path', type=str, default='/home/guizhewei/guizhewei/grasp_pose_dataset/unoptimized')
    parser.add_argument('--object_code', type=str, default=None, help='如果指定，只可视化该object_code的数据')
    parser.add_argument('--num', type=int, default=0, help='可视化第几个数据（如果指定了object_code）')
    parser.add_argument('--max_visualize', type=int, default=5, help='最多可视化多少个数据')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='使用的设备')
    args = parser.parse_args()

    device = args.device
    
    # 检查CUDA可用性
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        device = 'cpu'

    # 加载未优化的数据
    data_dict = np.load(os.path.join(args.result_path, args.grasp_file + '.npy'), allow_pickle=True)
    print(f"加载了 {len(data_dict)} 个数据条目")
    
    # 过滤数据
    filtered_data = []
    for data in data_dict:
        if args.object_code is not None and data['object_code'] != args.object_code:
            continue
        filtered_data.append(data)
    
    if len(filtered_data) == 0:
        print(f"错误: 没有找到匹配的数据 (object_code={args.object_code})")
        exit(1)
    
    print(f"找到 {len(filtered_data)} 个匹配的数据条目")
    
    # 限制可视化数量
    num_visualize = min(args.max_visualize, len(filtered_data))
    if args.object_code is not None:
        # 如果指定了object_code，只可视化指定的那个
        visualize_indices = [args.num] if args.num < len(filtered_data) else [0]
    else:
        visualize_indices = list(range(num_visualize))
    
    # 收集所有需要可视化的object_code和scale
    object_code_list = []
    scene_scale_list = []
    obj_scale_list = []
    hand_pose_list = []
    
    for idx in visualize_indices:
        data = filtered_data[idx]
        object_code_list.append(data['object_code'])
        
        # 读取 scene_scale 和 obj_scale
        scene_scale = None
        obj_scale = None
        if 'scene_scale' in data:
            scene_scale = data['scene_scale']
        if 'obj_scale' in data:
            obj_scale = data['obj_scale']
        
        scene_scale_list.append(scene_scale)
        obj_scale_list.append(obj_scale)
        
        # 构建hand_pose
        qpos = data['qpos']
        rot = data['hand_rot6d']
        hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
        hand_pose_list.append(hand_pose)
    
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
    
    # 使用读取的 scale 初始化 object_model
    object_model.initialize(object_code_list=object_code_list, scene_scale_list=scene_scale_list, obj_scale_list=obj_scale_list)
    
    # 打印 scale 信息
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
    
    # 可视化每个数据
    hand_pose_tensor = torch.stack(hand_pose_list, dim=0)
    hand_model.set_parameters(hand_pose_tensor)
    
    for i, idx in enumerate(visualize_indices):
        data = filtered_data[idx]
        print(f"\n可视化数据 {i+1}/{len(visualize_indices)}: object_code={data['object_code']}, idx={data.get('idx', 'N/A')}")
        
        hand_plotly = hand_model.get_plotly_data(i=i, opacity=1, color='lightblue', with_contact_points=False)
        object_plotly = object_model.get_plotly_data(i=i, color='lightgreen', opacity=1)
        
        fig = go.Figure(hand_plotly + object_plotly)
        
        # 添加标题和信息
        title = f'未优化数据 - Object: {data["object_code"]}'
        if 'idx' in data:
            title += f' (idx: {data["idx"]})'
        
        fig.update_layout(
            scene_aspectmode='data',
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        # 添加scale信息到注释
        scale_info = f'Index {i}'
        if scene_scale_list[i] is not None:
            scale_info += f'  scene_scale: {scene_scale_list[i]:.4f}'
        if obj_scale_list[i] is not None:
            if isinstance(obj_scale_list[i], np.ndarray):
                obj_scale_val = np.mean(obj_scale_list[i]) if obj_scale_list[i].ndim == 1 else obj_scale_list[i].flat[0]
            else:
                obj_scale_val = obj_scale_list[i]
            scale_info += f'  obj_scale: {obj_scale_val:.4f}'
        if scene_scale_list[i] is not None and obj_scale_list[i] is not None:
            if isinstance(obj_scale_list[i], np.ndarray):
                obj_scale_val = np.mean(obj_scale_list[i]) if obj_scale_list[i].ndim == 1 else obj_scale_list[i].flat[0]
            else:
                obj_scale_val = obj_scale_list[i]
            total_scale = float(scene_scale_list[i]) * float(obj_scale_val)
            scale_info += f'  total_scale: {total_scale:.4f}'
        
        fig.add_annotation(text=scale_info, x=0.5, y=0.05, xref='paper', yref='paper')
        
        fig.show()
        
        if i < len(visualize_indices) - 1:
            input(f"按 Enter 键继续可视化下一个数据 ({i+2}/{len(visualize_indices)})...")
    
    print(f"\n完成！共可视化了 {len(visualize_indices)} 个数据")

