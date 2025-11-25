#!/usr/bin/env python3
"""
将1027_6000实验目录下的所有优化结果npy文件转换回dexycb格式的完整npy文件
"""

import numpy as np
import os
import sys
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as transform

# 尝试导入sapien（可选，用于加载包含Pose对象的npy文件）
try:
    import sapien
    HAS_SAPIEN = True
except ImportError:
    HAS_SAPIEN = False
    print("警告: 未安装sapien库，如果原始数据文件包含Pose对象，可能需要安装sapien")

# 定义关节名称
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

def quaternion_to_rotation_matrix(quaternion):
    """将四元数转换为旋转矩阵"""
    rotation = R.from_quat(quaternion)
    return rotation.as_matrix()

def object_pose_to_matrix(position, quaternion):
    """
    将物体pose (位置和四元数)转换为4x4变换矩阵
    
    参数:
    - position: (3,) numpy.ndarray, 位置 [x, y, z]
    - quaternion: (4,) numpy.ndarray, [w, x, y, z] 四元数 (wxyz格式)
    
    返回:
    - transformation_matrix: (4, 4) numpy.ndarray, 对应的变换矩阵
    """
    # wxyz -> xyzw
    quaternion = np.concatenate([quaternion[1:4], quaternion[0:1]])
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    return transformation_matrix

def get_global_pose_for_training(global_grasp_poses, oc_hand_pose, obj_idx):
    """
    从object-centric的手部pose转换为world frame的pose
    
    参数:
    - global_grasp_poses: 原始的dexycb格式数据字典
    - oc_hand_pose: object-centric的手部pose [trans(3) + euler(3)]
    - obj_idx: 物体索引
    
    返回:
    - hand_pose_world: world frame中的手部pose [trans(3) + euler(3)]
    """
    # 手部pose: 平移 + 欧拉角
    oc_hand_trans = oc_hand_pose[:3]
    oc_hand_orient = oc_hand_pose[3:]

    # 物体pose
    obj_pos = global_grasp_poses[obj_idx]['target_pose_world'][0].p
    obj_quat = global_grasp_poses[obj_idx]['target_pose_world'][0].q
    object_pose = object_pose_to_matrix(obj_pos, obj_quat)
    W_T_O = object_pose

    # 从object-centric pose构建 O_T_H
    R_oh = R.from_euler('xyz', oc_hand_orient, degrees=False).as_matrix()
    O_T_H = np.eye(4)
    O_T_H[:3, :3] = R_oh
    O_T_H[:3, 3] = np.asarray(oc_hand_trans)

    # 转换回world: W_T_H = W_T_O @ O_T_H
    W_T_H = W_T_O @ O_T_H

    # 提取
    R_wh = W_T_H[:3, :3]
    t_wh = W_T_H[:3, 3]
    euler_wh = R.from_matrix(R_wh).as_euler('XYZ', degrees=False)

    # Hand pose in world frame
    hand_pose_world = t_wh.tolist() + euler_wh.tolist()

    return hand_pose_world

def main():
    import argparse
    parser = argparse.ArgumentParser(description='将优化结果转换为dexycb格式')
    parser.add_argument('--result_dir', 
                       default='/home/guizhewei/guizhewei/DexGraspNet/data/experiments/1103_6000_dexycb/results',
                       help='优化结果文件目录')
    parser.add_argument('--global_grasp_poses_path',
                       default='/home/guizhewei/guizhewei/Dexycb_dataset/grasp_poses_1030_1518_new.npy',
                       help='原始dexycb数据文件路径')
    parser.add_argument('--output_path',
                       default='/home/guizhewei/guizhewei/grasp_pose_dataset/optimized/grasp_poses_1103_6000_dexycb.npy',
                       help='输出文件路径')
    args = parser.parse_args()
    
    result_dir = args.result_dir
    global_grasp_poses_path = args.global_grasp_poses_path
    output_path = args.output_path
    
    # 检查路径是否存在
    if not os.path.exists(result_dir):
        raise FileNotFoundError(f"结果目录不存在: {result_dir}")
    
    # 加载原始的global_grasp_poses数据
    print(f"正在加载原始dexycb数据: {global_grasp_poses_path}")
    if not os.path.exists(global_grasp_poses_path):
        print(f"错误: 原始dexycb数据文件不存在: {global_grasp_poses_path}")
        print("请使用 --global_grasp_poses_path 参数指定正确的路径")
        sys.exit(1)
    
    try:
        global_grasp_poses_dict = np.load(global_grasp_poses_path, allow_pickle=True).item()
        print(f"成功加载，包含 {len(global_grasp_poses_dict)} 个条目")
    except ModuleNotFoundError as e:
        if 'sapien' in str(e).lower():
            print(f"错误: 需要sapien库来加载包含Pose对象的数据文件")
            print("请安装sapien: pip install sapien")
            print("或者使用已经加载并保存的纯numpy格式数据")
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"加载原始dexycb数据失败: {e}")
        print("如果文件包含Pose对象，可能需要安装sapien库")
        sys.exit(1)
    
    # 读取所有优化结果文件
    print(f"\n正在读取优化结果文件: {result_dir}")
    npy_files = sorted([f for f in os.listdir(result_dir) if f.endswith('.npy')])
    print(f"找到 {len(npy_files)} 个npy文件")
    
    # 重建字典
    refine_dict = {}
    e_pen_list = []
    
    # 关节映射索引 (从notebook中的origin_map_idx逆映射)
    origin_map_idx = [0, 5, 10, 15, 18, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 4, 9, 14]
    inv_map_idx = [0] * len(origin_map_idx)
    for new_pos, old_pos in enumerate(origin_map_idx):
        inv_map_idx[old_pos] = new_pos
    
    for fname in npy_files:
        try:
            # 从文件名提取索引和物体名称
            name_no_ext = os.path.splitext(fname)[0]
            parts = name_no_ext.split('_', 1)
            if len(parts) != 2:
                print(f"警告: 跳过文件名格式不正确的文件: {fname}")
                continue
            
            object_idx = int(parts[0])
            object_code = parts[1]
            
            # 检查object_idx是否在原始数据中
            if object_idx not in global_grasp_poses_dict:
                print(f"警告: 索引 {object_idx} 不在原始数据中，跳过: {fname}")
                continue
            
            # 加载优化结果
            opt_data_path = os.path.join(result_dir, fname)
            opt_data = np.load(opt_data_path, allow_pickle=True)
            
            # 获取第一个batch的数据
            if isinstance(opt_data, np.ndarray):
                opt_data_dict = opt_data.item() if opt_data.ndim == 0 else opt_data[0]
            else:
                opt_data_dict = opt_data
            
            if not isinstance(opt_data_dict, dict):
                print(f"警告: {fname} 数据格式不正确，跳过")
                continue
            
            # 提取qpos
            hand_qpos = opt_data_dict.get('qpos', {})
            if not hand_qpos:
                print(f"警告: {fname} 中没有qpos数据，跳过")
                continue
            
            E_pen = opt_data_dict.get('E_pen', float('inf'))
            e_pen_list.append([E_pen, object_idx])
            
            # 提取object-centric的手部pose
            hand_trans = np.array([hand_qpos[name] for name in translation_names])
            hand_rot = np.array([hand_qpos[name] for name in rot_names])
            oc_hand_pose = np.concatenate([hand_trans, hand_rot])
            
            # 转换为global pose
            global_pose = get_global_pose_for_training(
                global_grasp_poses_dict, oc_hand_pose, object_idx
            )
            
            # 提取手指关节角度
            finger_joints = [hand_qpos[name] for name in joint_names]
            # 应用逆映射
            mapped_finger_joints = [finger_joints[i] for i in inv_map_idx]
            
            # 构建robot_pose数组: [trans(3) + euler(3) + joints(19)]
            robot_pose = np.array(global_pose + mapped_finger_joints, dtype=np.float32)
            
            # 复制原始数据并更新robot_pose
            refine_dict[object_idx] = global_grasp_poses_dict[object_idx].copy()
            refine_dict[object_idx]['robot_pose'] = [robot_pose]
            
            print(f"处理完成: idx={object_idx}, object={object_code}, E_pen={E_pen:.6f}")
            
        except Exception as e:
            print(f"处理 {fname} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 按E_pen排序输出
    e_pen_list = sorted(e_pen_list, key=lambda x: x[0])
    print("\n能量统计 (按E_pen排序):")
    for e_pen, obj_idx in e_pen_list[:10]:  # 只显示前10个
        print(f"  object_idx: {obj_idx}, E_pen: {e_pen:.6f}")
    
    # 保存结果
    print(f"\n正在保存结果到: {output_path}")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    np.save(output_path, refine_dict, allow_pickle=True)
    print(f"成功保存 {len(refine_dict)} 个条目到 {output_path}")
    
    # 验证保存的数据
    print("\n验证保存的数据...")
    loaded_dict = np.load(output_path, allow_pickle=True).item()
    print(f"验证成功: 加载了 {len(loaded_dict)} 个条目")
    
    # 显示一个示例
    if loaded_dict:
        first_key = sorted(loaded_dict.keys())[0]
        example = loaded_dict[first_key]
        print(f"\n示例条目 (idx={first_key}):")
        print(f"  target_object_name: {example.get('target_object_name', 'N/A')}")
        print(f"  target_object_idx: {example.get('target_object_idx', 'N/A')}")
        print(f"  robot_pose shape: {example.get('robot_pose', [None])[0].shape if example.get('robot_pose') else 'N/A'}")
        print(f"  robot_names: {example.get('robot_names', 'N/A')}")
        print(f"  hand_type: {example.get('hand_type', 'N/A')}")

if __name__ == '__main__':
    main()

