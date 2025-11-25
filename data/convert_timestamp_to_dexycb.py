#!/usr/bin/env python3
"""
将时间戳命名格式（如20200903_105205_008_pudding_box.npy）的优化结果npy文件转换回dexycb格式的完整npy文件
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

def get_global_pose_for_training(global_grasp_poses, oc_hand_pose, timestamp_key):
    """
    从object-centric的手部pose转换为world frame的pose
    
    参数:
    - global_grasp_poses: 原始的dexycb格式数据字典
    - oc_hand_pose: object-centric的手部pose [trans(3) + euler(3)]
    - timestamp_key: 时间戳键（字符串格式，如'20200903_105205'）
    
    返回:
    - hand_pose_world: world frame中的手部pose [trans(3) + euler(3)]
    """
    # 手部pose: 平移 + 欧拉角
    oc_hand_trans = oc_hand_pose[:3]
    oc_hand_orient = oc_hand_pose[3:]

    # 物体pose
    obj_pos = global_grasp_poses[timestamp_key]['target_pose_world'][0].p
    obj_quat = global_grasp_poses[timestamp_key]['target_pose_world'][0].q
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

def extract_timestamp_from_filename(filename):
    """
    从文件名提取时间戳
    
    文件名格式: 20200903_105205_008_pudding_box.npy
    返回: '20200903_105205'
    """
    name_no_ext = os.path.splitext(filename)[0]
    # 时间戳格式通常是 YYYYMMDD_HHMMSS
    # 在第一个下划线后，再找第二个下划线前的部分
    parts = name_no_ext.split('_')
    if len(parts) >= 2:
        # 时间戳应该是前两部分：20200903_105205
        timestamp = f"{parts[0]}_{parts[1]}"
        return timestamp
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='将时间戳命名格式的优化结果转换为dexycb格式')
    parser.add_argument('--result_dir', 
                       default='/home/guizhewei/guizhewei/DexGraspNet/data/experiments/dexycb_4_box/results',
                       help='优化结果文件目录')
    parser.add_argument('--global_grasp_poses_path',
                       default='/home/guizhewei/guizhewei/Dexycb_dataset/grasp_poses_dexycb_4.npy',
                       help='原始dexycb数据文件路径')
    parser.add_argument('--output_path',
                       default='/home/guizhewei/guizhewei/grasp_pose_dataset/optimized/grasp_poses_timestamp_dexycb.npy',
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
        # 显示一些键的示例
        sample_keys = list(global_grasp_poses_dict.keys())[:5]
        print(f"示例键: {sample_keys}")
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
            
            # 从优化结果中获取idx（时间戳）
            timestamp_key = opt_data_dict.get('idx')
            if timestamp_key is None:
                # 如果优化结果中没有idx，尝试从文件名提取
                timestamp_key = extract_timestamp_from_filename(fname)
                if timestamp_key is None:
                    print(f"警告: {fname} 中没有idx字段且无法从文件名提取时间戳，跳过")
                    continue
            
            # 检查timestamp_key是否在原始数据中
            if timestamp_key not in global_grasp_poses_dict:
                print(f"警告: 时间戳 {timestamp_key} 不在原始数据中，跳过: {fname}")
                # 尝试检查是否是整数键格式
                try:
                    int_key = int(timestamp_key)
                    if int_key in global_grasp_poses_dict:
                        timestamp_key = int_key
                        print(f"  找到整数键格式: {int_key}")
                    else:
                        continue
                except (ValueError, TypeError):
                    continue
            
            # 提取qpos
            hand_qpos = opt_data_dict.get('qpos', {})
            if not hand_qpos:
                print(f"警告: {fname} 中没有qpos数据，跳过")
                continue
            
            E_pen = opt_data_dict.get('E_pen', float('inf'))
            e_pen_list.append([E_pen, timestamp_key])
            
            # 提取object-centric的手部pose
            hand_trans = np.array([hand_qpos[name] for name in translation_names])
            hand_rot = np.array([hand_qpos[name] for name in rot_names])
            oc_hand_pose = np.concatenate([hand_trans, hand_rot])
            
            # 转换为global pose
            global_pose = get_global_pose_for_training(
                global_grasp_poses_dict, oc_hand_pose, timestamp_key
            )
            
            # 提取手指关节角度
            finger_joints = [hand_qpos[name] for name in joint_names]
            # 应用逆映射
            mapped_finger_joints = [finger_joints[i] for i in inv_map_idx]
            
            # 构建robot_pose数组: [trans(3) + euler(3) + joints(19)]
            robot_pose = np.array(global_pose + mapped_finger_joints, dtype=np.float32)
            
            # 复制原始数据并更新robot_pose
            refine_dict[timestamp_key] = global_grasp_poses_dict[timestamp_key].copy()
            refine_dict[timestamp_key]['robot_pose'] = [robot_pose]
            
            print(f"处理完成: timestamp={timestamp_key}, file={fname}, E_pen={E_pen:.6f}")
            
        except Exception as e:
            print(f"处理 {fname} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 按E_pen排序输出
    e_pen_list = sorted(e_pen_list, key=lambda x: x[0])
    print("\n能量统计 (按E_pen排序):")
    for e_pen, timestamp_key in e_pen_list[:10]:  # 只显示前10个
        print(f"  timestamp: {timestamp_key}, E_pen: {e_pen:.6f}")
    
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
        print(f"\n示例条目 (timestamp={first_key}):")
        print(f"  target_object_name: {example.get('target_object_name', 'N/A')}")
        print(f"  target_object_idx: {example.get('target_object_idx', 'N/A')}")
        print(f"  robot_pose shape: {example.get('robot_pose', [None])[0].shape if example.get('robot_pose') else 'N/A'}")
        print(f"  robot_names: {example.get('robot_names', 'N/A')}")
        print(f"  hand_type: {example.get('hand_type', 'N/A')}")

if __name__ == '__main__':
    main()

