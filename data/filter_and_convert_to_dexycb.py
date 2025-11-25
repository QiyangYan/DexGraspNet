#!/usr/bin/env python3
"""
根据能量阈值过滤抓取姿态，并将结果转换为 DexYCB 格式的数据文件。
"""

import argparse
import copy
import importlib
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# 尝试导入 sapien（可选，用于处理包含 Pose 对象的数据）
sapien_spec = importlib.util.find_spec("sapien")
if sapien_spec is not None:
    sapien = importlib.import_module("sapien")  # noqa: F401
    HAS_SAPIEN = True
else:
    HAS_SAPIEN = False
    sapien = None

# 关节名称定义
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

# 关节顺序映射（来自 convert_1027_6000_to_dexycb.py）
origin_map_idx = [0, 5, 10, 15, 18, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 4, 9, 14]
inv_map_idx = [0] * len(origin_map_idx)
for new_pos, old_pos in enumerate(origin_map_idx):
    inv_map_idx[old_pos] = new_pos


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """将四元数转换为旋转矩阵，输入为 xyzw。"""
    rotation = R.from_quat(quaternion)
    return rotation.as_matrix()


def object_pose_to_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    """
    将物体位姿（位置 + 四元数 wxyz）转换为 4x4 变换矩阵。
    """
    quaternion_xyzw = np.concatenate([quaternion_wxyz[1:4], quaternion_wxyz[0:1]])
    rotation_matrix = quaternion_to_rotation_matrix(quaternion_xyzw)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    return transformation_matrix


def get_global_pose_for_training(global_grasp_poses: Dict[int, Dict], oc_hand_pose: np.ndarray, obj_idx: int) -> List[float]:
    """
    从 object-centric 的手部位姿获取 world frame 下的手部位姿。
    返回 [trans(3) + euler(3)]。
    """
    oc_hand_trans = oc_hand_pose[:3]
    oc_hand_orient = oc_hand_pose[3:]

    obj_entry = global_grasp_poses[obj_idx]
    obj_pos = obj_entry['target_pose_world'][0].p
    obj_quat = obj_entry['target_pose_world'][0].q
    object_pose = object_pose_to_matrix(obj_pos, obj_quat)

    R_oh = R.from_euler('xyz', oc_hand_orient, degrees=False).as_matrix()
    O_T_H = np.eye(4)
    O_T_H[:3, :3] = R_oh
    O_T_H[:3, 3] = np.asarray(oc_hand_trans)

    W_T_H = object_pose @ O_T_H

    R_wh = W_T_H[:3, :3]
    t_wh = W_T_H[:3, 3]
    euler_wh = R.from_matrix(R_wh).as_euler('XYZ', degrees=False)

    return t_wh.tolist() + list(euler_wh)


def iter_pose_entries(np_data: np.ndarray) -> Iterable[Dict]:
    """
    遍历 np.load 读取的结果，返回里面的 dict 条目。
    """
    if isinstance(np_data, np.ndarray):
        if np_data.dtype == object:
            data_list = np_data.tolist()
        else:
            data_list = np_data
    elif isinstance(np_data, list):
        data_list = np_data
    else:
        data_list = [np_data]

    if isinstance(data_list, dict):
        yield data_list
        return

    if not isinstance(data_list, (list, tuple)):
        return

    for item in data_list:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, (list, tuple)):
            for sub_item in item:
                if isinstance(sub_item, dict):
                    yield sub_item


def pose_passes_filters(entry: Dict, max_e_pen: float, max_e_spen: float, max_e_cmap: float) -> bool:
    """
    判断该条目是否满足能量阈值过滤条件。
    缺失某项能量视为不通过。
    """
    try:
        e_pen = float(entry['E_pen'])
        e_spen = float(entry['E_spen'])
        e_cmap = float(entry['E_cmap'])
    except (KeyError, TypeError, ValueError):
        return False

    if e_pen > max_e_pen:
        return False
    if e_spen > max_e_spen:
        return False
    if e_cmap > max_e_cmap:
        return False
    return True


def convert_candidate_pose(global_grasp_poses: Dict[int, Dict], object_idx: int, hand_qpos: Dict) -> np.ndarray:
    """
    将 object-centric 的 hand_qpos 转换为 world frame 的 robot_pose。
    """
    hand_trans = np.array([hand_qpos[name] for name in translation_names], dtype=np.float32)
    hand_rot = np.array([hand_qpos[name] for name in rot_names], dtype=np.float32)
    oc_hand_pose = np.concatenate([hand_trans, hand_rot], axis=0)

    global_pose = get_global_pose_for_training(global_grasp_poses, oc_hand_pose, object_idx)

    finger_joints = [hand_qpos[name] for name in joint_names]
    mapped_finger_joints = [finger_joints[i] for i in inv_map_idx]

    robot_pose = np.array(global_pose + mapped_finger_joints, dtype=np.float32)
    return robot_pose


def select_best_candidates(entries: List[Tuple[Dict, np.ndarray]], keep_best_only: bool) -> List[Tuple[Dict, np.ndarray]]:
    """
    根据能量从小到大排序，并决定是否只保留最优解。
    排序权重优先级：E_pen -> E_cmap -> E_spen。
    """
    if not entries:
        return []

    sorted_entries = sorted(
        entries,
        key=lambda item: (
            float(item[0]['E_pen']),
            float(item[0]['E_cmap']),
            float(item[0]['E_spen'])
        )
    )

    if keep_best_only:
        return [sorted_entries[0]]
    return sorted_entries


def collect_npy_files(root_dir: str) -> List[Tuple[str, str]]:
    """
    递归遍历目录，收集所有 .npy 文件。
    返回列表元素为 (绝对路径, 相对 root_dir 路径)。
    """
    collected = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith('.npy'):
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root_dir)
            collected.append((full_path, rel_path))
    collected.sort(key=lambda item: item[1])
    return collected


def resolve_object_key(global_grasp_poses: Dict, primary_key, *fallbacks):
    """
    根据候选键在 global_grasp_poses 中查找匹配项。
    支持 int、numpy 数值类型、字符串互相转换。
    """
    candidates = [primary_key] + list(fallbacks)
    for candidate in candidates:
        if candidate is None:
            continue
        key = candidate
        if isinstance(key, np.generic):
            key = key.item()

        # 直接匹配
        if key in global_grasp_poses:
            return key

        # 字符串互转
        if isinstance(key, str):
            stripped = key.strip()
            if stripped in global_grasp_poses:
                return stripped
            try:
                int_val = int(stripped)
            except ValueError:
                int_val = None
            if int_val is not None and int_val in global_grasp_poses:
                return int_val
        else:
            try:
                int_val = int(key)
            except (ValueError, TypeError):
                int_val = None
            if int_val is not None and int_val in global_grasp_poses:
                return int_val

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='根据能量阈值过滤抓取姿态并转换为 DexYCB 格式'
    )
    parser.add_argument(
        '--result_dir',
        default='/home/guizhewei/guizhewei/DexGraspNet/data/experiments/bodex_pen_100_spen_100_cmap_1000/results',
        help='优化结果所在目录（保存了 *.npy 抓取文件）'
    )
    parser.add_argument(
        '--global_grasp_poses_path',
        default='/home/guizhewei/guizhewei/Dexycb_dataset/grasp_poses_1030_1518_new.npy',
        help='原始 DexYCB 抓取数据文件路径'
    )
    parser.add_argument(
        '--output_path',
        default='/home/guizhewei/guizhewei/grasp_pose_dataset/optimized/grasp_poses_filtered_bodex.npy',
        help='过滤并转换后的输出路径'
    )
    parser.add_argument('--max_E_pen', type=float, default=0.1, help='E_pen 阈值（默认 0.1）')
    parser.add_argument('--max_E_spen', type=float, default=0.1, help='E_spen 阈值（默认 0.1）')
    parser.add_argument('--max_E_cmap', type=float, default=0.15, help='E_cmap 阈值（默认 0.15）')
    parser.add_argument('--keep_best_only', action='store_true', help='若指定，仅保留每个物体中能量最优的一个解')
    parser.add_argument('--verbose', action='store_true', help='输出更详细的日志')
    parser.add_argument('--dry_run', action='store_true', help='只打印过滤结果，不写入输出文件')
    return parser.parse_args()


def main():
    args = parse_args()

    result_dir = args.result_dir
    global_grasp_poses_path = args.global_grasp_poses_path
    output_path = args.output_path

    if not os.path.exists(result_dir):
        raise FileNotFoundError(f"结果目录不存在: {result_dir}")

    if not os.path.exists(global_grasp_poses_path):
        print(f"错误: 原始 DexYCB 数据文件不存在: {global_grasp_poses_path}")
        print("请通过 --global_grasp_poses_path 指定正确路径。")
        sys.exit(1)

    print(f"正在加载原始 DexYCB 数据: {global_grasp_poses_path}")
    try:
        global_grasp_poses_dict = np.load(global_grasp_poses_path, allow_pickle=True).item()
    except ModuleNotFoundError as exc:
        if 'sapien' in str(exc).lower():
            print("错误: 需要安装 sapien 库来读取包含 Pose 对象的文件。")
            print("请执行: pip install sapien")
            sys.exit(1)
        raise

    print(f"成功加载 {len(global_grasp_poses_dict)} 条原始抓取数据。")

    npy_files = collect_npy_files(result_dir)
    print(f"在结果目录中递归找到 {len(npy_files)} 个抓取文件。")

    refine_dict: Dict[int, Dict] = {}
    stats_total = 0
    stats_kept = 0
    stats_filtered = 0
    per_object_counts = defaultdict(lambda: {'total': 0, 'kept': 0})

    for full_path, rel_path in npy_files:
        fname = os.path.basename(full_path)
        name_no_ext = os.path.splitext(fname)[0]
        parts = name_no_ext.split('_', 1)
        if len(parts) != 2:
            print(f"警告: 跳过文件名格式不正确的文件: {fname}")
            continue

        try:
            fallback_idx = int(parts[0])
        except ValueError:
            fallback_idx = parts[0]

        fallback_code = parts[1]
        if fallback_idx not in global_grasp_poses_dict:
            resolved_fallback = resolve_object_key(global_grasp_poses_dict, fallback_idx)
            if resolved_fallback is None:
                print(f"警告: 无法通过文件名解析有效索引，跳过: {rel_path}")
                fallback_idx = None
            else:
                fallback_idx = resolved_fallback

        try:
            opt_data = np.load(full_path, allow_pickle=True)
        except Exception as exc:
            print(f"读取 {fname} 失败: {exc}")
            continue

        candidate_entries = list(iter_pose_entries(opt_data))
        if not candidate_entries:
            if args.verbose:
                print(f"{rel_path}: 未找到抓取条目。")
            continue

        filtered_candidates: List[Tuple[Dict, np.ndarray, object]] = []
        for entry in candidate_entries:
            stats_total += 1

            entry_object_code = entry.get('object_code', fallback_code)
            entry_idx_raw = entry.get('idx', entry.get('object_idx', fallback_idx))
            object_idx = resolve_object_key(global_grasp_poses_dict, entry_idx_raw, fallback_idx, str(entry_idx_raw) if entry_idx_raw is not None else None)

            if object_idx is None:
                stats_filtered += 1
                if args.verbose:
                    print(f"{rel_path}: 条目缺少有效 idx（原值={entry_idx_raw}），跳过。")
                continue

            per_object_counts[str(object_idx)]['total'] += 1

            if not pose_passes_filters(entry, args.max_E_pen, args.max_E_spen, args.max_E_cmap):
                stats_filtered += 1
                continue

            qpos = entry.get('qpos')
            if not isinstance(qpos, dict):
                stats_filtered += 1
                if args.verbose:
                    print(f"{rel_path}: 条目缺少 qpos，跳过。")
                continue

            try:
                robot_pose = convert_candidate_pose(global_grasp_poses_dict, object_idx, qpos)
            except Exception as exc:
                stats_filtered += 1
                print(f"{rel_path}: 转换抓取失败（idx={object_idx}）: {exc}")
                continue

            filtered_candidates.append((entry, robot_pose, object_idx))

        if not filtered_candidates:
            if args.verbose:
                print(f"{rel_path}: 无满足阈值的抓取。")
            continue

        grouped_by_idx = defaultdict(list)
        for entry, pose, obj_idx in filtered_candidates:
            grouped_by_idx[obj_idx].append((entry, pose))

        for object_idx, entry_list in grouped_by_idx.items():
            selected_candidates = select_best_candidates(entry_list, args.keep_best_only)
            per_object_counts[str(object_idx)]['kept'] += len(selected_candidates)
            stats_kept += len(selected_candidates)

            if object_idx not in refine_dict:
                base_entry = copy.deepcopy(global_grasp_poses_dict[object_idx])
                base_entry['robot_pose'] = []
                base_entry['robot_pose_energy'] = []
                base_entry['filtered_object_code'] = []
                base_entry['source_files'] = []
                refine_dict[object_idx] = base_entry

            target_entry = refine_dict[object_idx]
            target_entry['robot_pose'].extend([pose for _, pose in selected_candidates])
            target_entry['robot_pose_energy'].extend([
                {
                    'E_pen': float(entry['E_pen']),
                    'E_spen': float(entry['E_spen']),
                    'E_cmap': float(entry['E_cmap'])
                }
                for entry, _ in selected_candidates
            ])
            if entry_object_code and entry_object_code not in target_entry['filtered_object_code']:
                target_entry['filtered_object_code'].append(entry_object_code)
            if rel_path not in target_entry['source_files']:
                target_entry['source_files'].append(rel_path)

        if args.verbose:
            for object_idx, entry_list in grouped_by_idx.items():
                selected_candidates = select_best_candidates(entry_list, args.keep_best_only)
                energies_text = ", ".join(
                    f"(idx={object_idx}, E_pen={entry['E_pen']:.4f}, E_spen={entry['E_spen']:.4f}, E_cmap={entry['E_cmap']:.4f})"
                    for entry, _ in selected_candidates
                )
                print(f"{rel_path}: 保留 {len(selected_candidates)} 条抓取 -> {energies_text}")

    print("\n过滤统计：")
    print(f"  总抓取数量: {stats_total}")
    print(f"  通过过滤数量: {stats_kept}")
    print(f"  被过滤数量: {stats_filtered}")
    print(f"  最终输出对象数量: {len(refine_dict)}")

    if args.verbose:
        for obj_idx, count_dict in sorted(per_object_counts.items(), key=lambda item: item[0]):
            print(f"  Object {obj_idx}: total={count_dict['total']}, kept={count_dict['kept']}")

    if args.dry_run:
        print("Dry-run 已启用，不写出结果文件。")
        return

    if not refine_dict:
        print("警告: 过滤后没有任何抓取，未写入输出文件。")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_path, refine_dict, allow_pickle=True)
    print(f"\n已保存 {len(refine_dict)} 个对象的抓取到: {output_path}")

    try:
        loaded_verify = np.load(output_path, allow_pickle=True).item()
        print(f"验证成功: 重新加载得到 {len(loaded_verify)} 个对象。")
    except Exception as exc:
        print(f"验证加载失败: {exc}")


if __name__ == '__main__':
    if not HAS_SAPIEN:
        print("提示: 未检测到 sapien 库，如果原始文件中包含 Pose 对象，需要安装 sapien 后再运行。")
    main()

