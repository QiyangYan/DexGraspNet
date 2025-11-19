"""
Author: GUI Zhewei
Description: 专门用于可视化penetration SDF的脚本
可视化物体表面采样点及其到手部模型的SDF距离
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
    parser = argparse.ArgumentParser(description='可视化抓取的penetration SDF')
    parser.add_argument('--object_code', type=str, default='47_008_pudding_box', help='物体代码')
    parser.add_argument('--num', type=int, default=0, help='结果索引')
    parser.add_argument('--result_path', type=str, 
                       default='/home/guizhewei/guizhewei/DexGraspNet/data/experiments/1103_6000_dexycb/results',
                       help='结果路径')
    parser.add_argument('--show_hand', action='store_true', help='是否显示手部模型')
    parser.add_argument('--show_object_mesh', action='store_true', help='是否显示物体网格')
    parser.add_argument('--point_size', type=float, default=3, help='采样点大小')
    parser.add_argument('--sdf_min', type=float, default=-0.01, help='SDF颜色条最小值')
    parser.add_argument('--sdf_max', type=float, default=0.01, help='SDF颜色条最大值')
    parser.add_argument('--use_voxel_grid', action='store_true', help='使用体素网格而不是表面采样点')
    parser.add_argument('--voxel_resolution', type=int, default=32, help='体素网格分辨率（每个维度）')
    parser.add_argument('--show_inside_only', action='store_true', help='只显示内部点（穿透点）')
    parser.add_argument('--show_outside_only', action='store_true', help='只显示外部点')
    parser.add_argument('--use_initial_pose', action='store_true', help='使用初始姿态而不是优化后的姿态')
    parser.add_argument('--verify_object_sdf', action='store_true', help='验证物体自身的SDF（查看表面采样点到物体的距离，应接近0）')
    parser.add_argument('--hand_only', action='store_true', help='只加载和观察手部SDF，不需要物体')
    args = parser.parse_args()

    # TorchSDF需要CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("警告: CUDA不可用，但TorchSDF需要CUDA来计算距离")
        print("请确保CUDA可用或使用支持CUDA的环境")

    print(f"\n{'='*60}")
    if args.hand_only:
        print(f"只观察手部SDF")
    elif args.verify_object_sdf:
        print(f"验证物体SDF")
    else:
        print(f"可视化Penetration SDF")
    print(f"物体: {args.object_code}")
    print(f"结果索引: {args.num}")
    print(f"设备: {device}")
    print(f"{'='*60}\n")

    # 加载结果
    result_file = os.path.join(args.result_path, args.object_code + '.npy')
    if not os.path.exists(result_file):
        print(f"错误: 结果文件不存在: {result_file}")
        sys.exit(1)
        
    data_dict = np.load(result_file, allow_pickle=True)[args.num]
    qpos = data_dict['qpos']
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + 
                            [qpos[name] for name in joint_names], 
                            dtype=torch.float, device=device)
    
    # 如果有初始姿态，也加载
    hand_pose_st = None
    if 'qpos_st' in data_dict:
        qpos_st = data_dict['qpos_st']
        rot = np.array(transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot + 
                                    [qpos_st[name] for name in joint_names], 
                                    dtype=torch.float, device=device)

    # 临时切换到mjcf目录以加载网格
    current_dir = os.getcwd()
    os.chdir('/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf')

    # 初始化手部模型
    hand_model = HandModel(
        mjcf_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf/omnihand_tendon_contact.xml',
        mesh_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/franka_omnihand_mjcf',
        contact_points_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/contact_points_omnihand.json',
        penetration_points_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/penetration_points_omnihand.json',
        device=device
    )

    # 切换回原目录
    os.chdir(current_dir)

    # 初始化物体模型（只在需要时加载）
    object_model = None
    if not args.hand_only:
        object_model = ObjectModel(
            data_root_path='/home/guizhewei/guizhewei/Dexycb_dataset/models',
            batch_size_each=1,
            num_samples=2000,  # 表面采样点数量
            device=device
        )
        
        name_no_ext = os.path.splitext(args.object_code)[0]
        object_idx, object_code = name_no_ext.split("_", 1)
        object_model.initialize(object_code_list=object_code)
        object_model.object_scale_tensor = torch.tensor(1, dtype=torch.float, device=device).reshape(1, 1)

    # 选择使用的手部姿态
    if args.use_initial_pose:
        if hand_pose_st is not None:
            selected_pose = hand_pose_st
            pose_name = "初始姿态"
            print(f"\n使用初始姿态 (qpos_st) 进行SDF计算")
        else:
            print(f"\n警告: 没有找到初始姿态 (qpos_st)，将使用优化后的姿态")
            selected_pose = hand_pose
            pose_name = "优化后的姿态"
    else:
        selected_pose = hand_pose
        pose_name = "优化后的姿态"
        print(f"\n使用优化后的姿态 (qpos) 进行SDF计算")
    
    # 设置手部姿态
    hand_model.set_parameters(selected_pose.unsqueeze(0))

    # 根据模式选择采样点
    if args.hand_only:
        # 手部SDF模式：生成手部周围的体素网格来可视化手部SDF
        voxel_res = args.voxel_resolution if args.voxel_resolution else 32
        print(f"手部SDF模式：使用体素网格，分辨率: {voxel_res}^3 = {voxel_res**3} 个点")
        print(f"说明: 内部点SDF>0 (手内部), 外部点SDF<0 (手外部), 表面点SDF≈0 (手表面)")
        
        # 获取手部的bounding box - 使用contact candidates和penetration keypoints
        contact_candidates = hand_model.get_contact_candidates()[0].detach().cpu().numpy()
        penetration_keypoints = hand_model.get_penetraion_keypoints()[0].detach().cpu().numpy()
        hand_points = np.vstack([contact_candidates, penetration_keypoints])
        
        bbox_min = hand_points.min(axis=0)
        bbox_max = hand_points.max(axis=0)
        
        # 扩展bbox以更好地观察手部周围的SDF场
        bbox_margin = 0.05  # 5cm的边界
        bbox_min -= bbox_margin
        bbox_max += bbox_margin
        
        print(f"手部Bounding Box (扩展后):")
        print(f"  Min: [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]")
        print(f"  Max: [{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]")
        print(f"  Size: [{bbox_max[0]-bbox_min[0]:.4f}, {bbox_max[1]-bbox_min[1]:.4f}, {bbox_max[2]-bbox_min[2]:.4f}]")
        
        # 在bbox内生成均匀的体素网格
        x = np.linspace(bbox_min[0], bbox_max[0], voxel_res)
        y = np.linspace(bbox_min[1], bbox_max[1], voxel_res)
        z = np.linspace(bbox_min[2], bbox_max[2], voxel_res)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        voxel_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        voxel_points_tensor = torch.tensor(voxel_points, dtype=torch.float, device=device).unsqueeze(0)
        
        query_points = voxel_points_tensor
        query_points_np = voxel_points
        
    elif args.verify_object_sdf:
        # 验证物体SDF模式：使用体素网格覆盖整个bbox
        voxel_res = args.voxel_resolution if args.voxel_resolution else 32
        print(f"验证物体SDF模式：使用体素网格，分辨率: {voxel_res}^3 = {voxel_res**3} 个点")
        print(f"说明: 内部点SDF>0, 外部点SDF<0, 表面点SDF≈0")
        
        # 获取物体的bounding box
        object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
        object_vertices_scaled = (torch.tensor(object_model.object_mesh_list[0].vertices, 
                                              dtype=torch.float, device=device) * 
                                 object_model.object_scale_tensor[0, 0].item())
        
        bbox_min = object_vertices_scaled.min(dim=0)[0].cpu().numpy()
        bbox_max = object_vertices_scaled.max(dim=0)[0].cpu().numpy()
        
        print(f"物体Bounding Box:")
        print(f"  Min: [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]")
        print(f"  Max: [{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]")
        print(f"  Size: [{bbox_max[0]-bbox_min[0]:.4f}, {bbox_max[1]-bbox_min[1]:.4f}, {bbox_max[2]-bbox_min[2]:.4f}]")
        
        # 在bbox内生成均匀的体素网格
        x = np.linspace(bbox_min[0], bbox_max[0], voxel_res)
        y = np.linspace(bbox_min[1], bbox_max[1], voxel_res)
        z = np.linspace(bbox_min[2], bbox_max[2], voxel_res)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        voxel_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        voxel_points_tensor = torch.tensor(voxel_points, dtype=torch.float, device=device).unsqueeze(0)
        
        query_points = voxel_points_tensor
        query_points_np = voxel_points
        
    elif args.use_voxel_grid:
        print(f"使用体素网格模式，分辨率: {args.voxel_resolution}^3 = {args.voxel_resolution**3} 个点")
        
        # 获取物体的bounding box
        object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
        object_vertices_scaled = (torch.tensor(object_model.object_mesh_list[0].vertices, 
                                              dtype=torch.float, device=device) * 
                                 object_model.object_scale_tensor[0, 0].item())
        
        bbox_min = object_vertices_scaled.min(dim=0)[0].cpu().numpy()
        bbox_max = object_vertices_scaled.max(dim=0)[0].cpu().numpy()
        
        print(f"物体Bounding Box:")
        print(f"  Min: [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]")
        print(f"  Max: [{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]")
        print(f"  Size: [{bbox_max[0]-bbox_min[0]:.4f}, {bbox_max[1]-bbox_min[1]:.4f}, {bbox_max[2]-bbox_min[2]:.4f}]")
        
        # 在bbox内生成均匀的体素网格
        x = np.linspace(bbox_min[0], bbox_max[0], args.voxel_resolution)
        y = np.linspace(bbox_min[1], bbox_max[1], args.voxel_resolution)
        z = np.linspace(bbox_min[2], bbox_max[2], args.voxel_resolution)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        voxel_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        voxel_points_tensor = torch.tensor(voxel_points, dtype=torch.float, device=device).unsqueeze(0)
        
        query_points = voxel_points_tensor
        query_points_np = voxel_points
    else:
        print(f"使用表面采样点模式，采样点数量: {object_model.num_samples}")
        
        # 使用物体表面采样点
        object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
        object_surface_points_scaled = object_model.surface_points_tensor * object_scale
        
        query_points = object_surface_points_scaled
        query_points_np = object_surface_points_scaled[0].detach().cpu().numpy()
    
    try:
        # 根据模式选择计算哪个SDF
        if args.hand_only:
            # 计算每个点到手部的SDF距离
            distances = hand_model.cal_distance(query_points)
        elif args.verify_object_sdf:
            # 计算每个点到物体自身的SDF距离
            distances_full, _ = object_model.cal_distance(query_points)
            distances = distances_full
        else:
            # 计算每个点到手部的SDF距离
            distances = hand_model.cal_distance(query_points)
        
        # 转换到CPU用于可视化
        distances_np = distances[0].detach().cpu().numpy()
        
        # 打印统计信息
        n_penetrating = (distances_np > 0).sum()
        n_outside = (distances_np <= 0).sum()
        max_value = distances_np.max()
        min_value = distances_np.min()
        mean_value = distances_np.mean()
        std_value = distances_np.std()
        abs_mean = np.abs(distances_np).mean()
        abs_max = np.abs(distances_np).max()
        
        print(f"\n{'='*60}")
        if args.hand_only:
            print(f"手部SDF统计信息")
        elif args.verify_object_sdf:
            print(f"物体SDF验证统计")
        else:
            print(f"穿透统计信息")
        print(f"{'='*60}")
        print(f"总采样点数:        {len(distances_np)}")
        
        if args.hand_only:
            # 对于手部SDF（体素网格）
            n_near_zero = (np.abs(distances_np) < 0.001).sum()
            print(f"表面附近点数 (|SDF|<0.001): {n_near_zero} ({100*n_near_zero/len(distances_np):.2f}%)")
            print(f"内部点数 (SDF>0):  {n_penetrating} ({100*n_penetrating/len(distances_np):.2f}%)")
            print(f"外部点数 (SDF<0):  {n_outside} ({100*n_outside/len(distances_np):.2f}%)")
            print(f"\n说明: 手部bbox内的体素点分布:")
            print(f"  - 内部点(SDF>0): 在手部内部")
            print(f"  - 外部点(SDF<0): 在手部外部")
            print(f"  - 表面点(SDF≈0): 接近手部表面")
        elif args.verify_object_sdf:
            # 对于物体SDF验证（体素网格）
            n_near_zero = (np.abs(distances_np) < 0.001).sum()
            print(f"表面附近点数 (|SDF|<0.001): {n_near_zero} ({100*n_near_zero/len(distances_np):.2f}%)")
            print(f"内部点数 (SDF>0):  {n_penetrating} ({100*n_penetrating/len(distances_np):.2f}%)")
            print(f"外部点数 (SDF<0):  {n_outside} ({100*n_outside/len(distances_np):.2f}%)")
            print(f"\n说明: bbox内的体素点应该合理分布:")
            print(f"  - 内部点应该>0")
            print(f"  - 外部点应该<0")
            print(f"  - 表面点应该接近0")
        else:
            print(f"穿透点数 (SDF>0):  {n_penetrating} ({100*n_penetrating/len(distances_np):.2f}%)")
            print(f"外部点数 (SDF<=0): {n_outside} ({100*n_outside/len(distances_np):.2f}%)")
        
        print(f"{'='*60}")
        print(f"SDF距离统计:")
        print(f"  最大值:             {max_value:.6f}")
        print(f"  最小值:             {min_value:.6f}")
        print(f"  平均值:             {mean_value:.6f}")
        print(f"  标准差:             {std_value:.6f}")
        
        if args.verify_object_sdf:
            print(f"  绝对值平均:         {abs_mean:.6f}")
            print(f"  绝对值最大:         {abs_max:.6f}")
        
        print(f"{'='*60}")
        
        # 如果有能量信息，也打印出来
        if 'energy' in data_dict:
            E_pen = data_dict.get('E_pen', 0)
            print(f"\nE_pen (优化中的穿透能量): {E_pen:.6f}")
            print(f"注意: E_pen = -sum(distances[distances>0])")
            print(f"计算验证: -sum(distances[distances>0]) = {-distances_np[distances_np>0].sum():.6f}")
            print(f"{'='*60}\n")
        
        # 创建可视化数据
        plotly_data = []
        
        # 可选: 显示手部姿态（验证物体SDF时不显示手部，但手部SDF模式总是显示手部）
        if (args.show_hand or args.hand_only) and not args.verify_object_sdf:
            # 如果使用初始姿态，只显示初始姿态
            if args.use_initial_pose:
                if hand_pose_st is not None:
                    hand_model.set_parameters(hand_pose_st.unsqueeze(0))
                    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.7, color='lightcoral', 
                                                            with_contact_points=False)
                    plotly_data.extend(hand_plotly)
            else:
                # 显示优化后的姿态，如果有初始姿态也可以显示作为对比
                if hand_pose_st is not None:
                    hand_model.set_parameters(hand_pose_st.unsqueeze(0))
                    hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.3, color='lightgray', 
                                                                with_contact_points=False)
                    plotly_data.extend(hand_st_plotly)
                
                hand_model.set_parameters(hand_pose.unsqueeze(0))
                hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=0.7, color='lightblue', 
                                                            with_contact_points=False)
                plotly_data.extend(hand_en_plotly)
            
            # 恢复用于计算SDF的姿态
            hand_model.set_parameters(selected_pose.unsqueeze(0))
        
        # 可选: 显示物体网格（手部SDF模式不显示物体）
        if args.show_object_mesh and not args.hand_only:
            object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=0.3)
            plotly_data.extend(object_plotly)
        
        # 根据用户选项过滤点
        if args.show_inside_only:
            mask = distances_np > 0
            filtered_points = query_points_np[mask]
            filtered_distances = distances_np[mask]
            filter_name = "内部点（穿透）"
        elif args.show_outside_only:
            mask = distances_np <= 0
            filtered_points = query_points_np[mask]
            filtered_distances = distances_np[mask]
            filter_name = "外部点"
        else:
            filtered_points = query_points_np
            filtered_distances = distances_np
            filter_name = "所有点"
        
        print(f"\n可视化: {filter_name}, 共 {len(filtered_points)} 个点")
        
        # 根据模式设置可视化
        if args.hand_only:
            # 手部SDF模式: 显示内部/表面/外部三种点
            # 内部(SDF>0.001)=红色, 表面(-0.001<SDF<0.001)=绿色, 外部(SDF<-0.001)=蓝色
            colors = []
            status_text = []
            for d in filtered_distances:
                if d > 0:
                    colors.append('red')
                    status_text.append('手内部')
                elif d < 0:
                    colors.append('lightblue')
                    status_text.append('手外部')
                # else:
                #     colors.append('green')
                #     status_text.append('手表面')
            
            points_plotly = go.Scatter3d(
                x=filtered_points[:, 0],
                y=filtered_points[:, 1],
                z=filtered_points[:, 2],
                mode='markers',
                marker=dict(
                    size=args.point_size,
                    color=colors,
                    line=dict(width=0),
                    opacity=0.6
                ),
                name='体素网格点',
                hovertemplate='<b>体素点</b><br>' +
                             '坐标: (%{x:.4f}, %{y:.4f}, %{z:.4f})<br>' +
                             'SDF: %{text:.6f}<br>' +
                             '位置: %{customdata}<br>' +
                             '<extra></extra>',
                text=filtered_distances,
                customdata=status_text
            )
        elif args.verify_object_sdf:
            # 物体SDF验证模式: 显示内部/表面/外部三种点
            # 内部(SDF>0.001)=红色, 表面(-0.001<SDF<0.001)=绿色, 外部(SDF<-0.001)=蓝色
            colors = []
            status_text = []
            for d in filtered_distances:
                if d > 0:
                    colors.append('red')
                    status_text.append('内部')
                elif d < 0:
                    colors.append('lightblue')
                    status_text.append('外部')
                # else:
                #     colors.append('green')
                #     status_text.append('表面')
            
            points_plotly = go.Scatter3d(
                x=filtered_points[:, 0],
                y=filtered_points[:, 1],
                z=filtered_points[:, 2],
                mode='markers',
                marker=dict(
                    size=args.point_size,
                    color=colors,
                    line=dict(width=0),
                    opacity=0.6
                ),
                name='体素网格点',
                hovertemplate='<b>体素点</b><br>' +
                             '坐标: (%{x:.4f}, %{y:.4f}, %{z:.4f})<br>' +
                             'SDF: %{text:.6f}<br>' +
                             '类型: %{customdata}<br>' +
                             '<extra></extra>',
                text=filtered_distances,
                customdata=status_text
            )
        elif args.use_voxel_grid:
            # 简单的二值颜色: 红色=内部(穿透), 浅绿色=外部
            colors = ['red' if d > 0 else 'lightgreen' for d in filtered_distances]
            
            # 主要内容: 显示体素点
            points_plotly = go.Scatter3d(
                x=filtered_points[:, 0],
                y=filtered_points[:, 1],
                z=filtered_points[:, 2],
                mode='markers',
                marker=dict(
                    size=args.point_size,
                    color=colors,
                    line=dict(width=0),
                    opacity=0.6
                ),
                name='体素网格点',
                hovertemplate='<b>体素点</b><br>' +
                             '坐标: (%{x:.4f}, %{y:.4f}, %{z:.4f})<br>' +
                             '状态: %{text}<br>' +
                             '<extra></extra>',
                text=['内部(穿透)' if d > 0 else '外部' for d in filtered_distances]
            )
        else:
            # 表面采样点模式: 使用连续颜色映射
            points_plotly = go.Scatter3d(
                x=filtered_points[:, 0],
                y=filtered_points[:, 1],
                z=filtered_points[:, 2],
                mode='markers',
                marker=dict(
                    size=args.point_size,
                    color=filtered_distances,
                    colorscale='RdYlGn_r',  # 红色=正值(穿透), 绿色=负值(外部)
                    colorbar=dict(
                        title='SDF值<br>(正=穿透)',
                        thickness=20,
                        len=0.7
                    ),
                    cmin=args.sdf_min,
                    cmax=args.sdf_max,
                    showscale=True,
                    line=dict(width=0)
                ),
                name='表面采样点',
                hovertemplate='<b>采样点</b><br>' +
                             '坐标: (%{x:.4f}, %{y:.4f}, %{z:.4f})<br>' +
                             'SDF距离: %{marker.color:.6f}<br>' +
                             '<extra></extra>'
            )
        
        plotly_data.append(points_plotly)
        
        # 创建图形
        fig = go.Figure(plotly_data)
        
        # 添加标题和注释
        if args.hand_only:
            voxel_res = args.voxel_resolution if args.voxel_resolution else 32
            title_text = f'手部SDF可视化 (体素网格 {voxel_res}³) [{pose_name}] - {args.object_code} (#{args.num})'
            n_inside = (distances_np > 0.001).sum()
            n_surface = (np.abs(distances_np) <= 0.001).sum()
            n_outside = (distances_np < -0.001).sum()
            subtitle = f'手内部: {n_inside} | 手表面: {n_surface} | 手外部: {n_outside} (红/绿/蓝)'
            fig.add_annotation(
                text=subtitle,
                xref='paper', yref='paper',
                x=0.5, y=0.02,
                showarrow=False,
                font=dict(size=12),
                bgcolor='rgba(255,255,255,0.8)'
            )
        elif args.verify_object_sdf:
            voxel_res = args.voxel_resolution if args.voxel_resolution else 32
            title_text = f'物体SDF验证 (体素网格 {voxel_res}³) - {args.object_code} (#{args.num})'
            n_inside = (distances_np > 0.001).sum()
            n_surface = (np.abs(distances_np) <= 0.001).sum()
            n_outside = (distances_np < -0.001).sum()
            subtitle = f'内部: {n_inside} | 表面: {n_surface} | 外部: {n_outside} (红/绿/蓝)'
            fig.add_annotation(
                text=subtitle,
                xref='paper', yref='paper',
                x=0.5, y=0.02,
                showarrow=False,
                font=dict(size=12),
                bgcolor='rgba(255,255,255,0.8)'
            )
        else:
            mode_text = f"体素网格 {args.voxel_resolution}³" if args.use_voxel_grid else "表面采样"
            title_text = f'Penetration SDF可视化 ({mode_text}) [{pose_name}] - {args.object_code} (#{args.num})'
            if 'energy' in data_dict:
                E_pen = data_dict.get('E_pen', 0)
                subtitle = f'E_pen: {E_pen:.5f} | 穿透点: {n_penetrating}/{len(distances_np)} ({100*n_penetrating/len(distances_np):.1f}%)'
                fig.add_annotation(
                    text=subtitle,
                    xref='paper', yref='paper',
                    x=0.5, y=0.02,
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor='rgba(255,255,255,0.8)'
                )
        
        fig.update_layout(
            title=title_text,
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=1200,
            height=900
        )
        
        print("正在打开可视化窗口...")
        fig.show()
        
    except Exception as e:
        print(f"\n错误: 无法计算SDF距离: {e}")
        print("请确保:")
        print("  1. CUDA可用 (TorchSDF需要CUDA)")
        print("  2. 手部模型和物体模型正确加载")
        import traceback
        traceback.print_exc()
        sys.exit(1)

