"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: Entry of the program, generate small-scale experiments for OmniHand
"""

import os

os.chdir(os.path.dirname(__file__))

import argparse
import json
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
import transforms3d
import plotly.graph_objects as go

from utils.hand_model import HandModel, SHADOWHAND_PART_NAMES
from utils.object_model import ObjectModel
from utils.energy import cal_energy
from utils.optimizer import Annealing, Adam
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

shadow_joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]

CONTACT_MAP_COLORSCALE = [
    [0.0, 'rgb(0, 0, 255)'],
    [0.25, 'rgb(0, 255, 255)'],
    [0.5, 'rgb(0, 255, 0)'],
    [0.75, 'rgb(255, 255, 0)'],
    [1.0, 'rgb(255, 0, 0)'],
]


def _normalize_contact_values(values):
    """
    将 contact value 归一化到 [0, 1]，同时返回原始最小值和最大值
    """
    contact_array = np.asarray(values, dtype=np.float32).reshape(-1)
    if contact_array.size == 0:
        return contact_array, 0.0, 0.0
    min_val = float(np.min(contact_array))
    max_val = float(np.max(contact_array))
    denom = max_val - min_val
    if denom < 1e-8:
        normalized = np.zeros_like(contact_array)
    else:
        normalized = (contact_array - min_val) / denom
    return normalized, min_val, max_val


def visualize_contact_map_with_custom_colors(hand_model, object_pc, contact_values,
                                             sample_idx=0, show_hand=True,
                                             marker_size=4.0, marker_opacity=0.85,
                                             title_prefix='Contact Map Visualization'):
    """
    使用与 visualize_dexonomy_grasp --show_contact_map 相同的颜色映射可视化 contact map
    """
    object_points = np.asarray(object_pc, dtype=np.float32)
    contact_values_array = np.asarray(contact_values, dtype=np.float32).reshape(-1)
    if object_points.shape[0] != contact_values_array.shape[0]:
        raise ValueError(f"object_pc size ({object_points.shape[0]}) 与 contact_values "
                         f"长度 ({contact_values_array.shape[0]}) 不匹配")

    normalized_values, min_val, max_val = _normalize_contact_values(contact_values_array)
    tick_positions = np.linspace(0.0, 1.0, 5)
    if max_val - min_val < 1e-8:
        tick_texts = [f"{min_val:.3f}"] * len(tick_positions)
    else:
        tick_texts = [f"{min_val + (max_val - min_val) * t:.3f}" for t in tick_positions]

    point_trace = go.Scatter3d(
        x=object_points[:, 0],
        y=object_points[:, 1],
        z=object_points[:, 2],
        mode='markers',
        marker=dict(
            size=marker_size,
            opacity=marker_opacity,
            color=normalized_values,
            cmin=0.0,
            cmax=1.0,
            colorscale=CONTACT_MAP_COLORSCALE,
            showscale=True,
            colorbar=dict(
                title='Contact value',
                tickvals=tick_positions,
                ticktext=tick_texts
            )
        ),
        name='Object Contact Map'
    )

    figure_data = [point_trace]
    if show_hand:
        figure_data.extend(
            hand_model.get_plotly_data(
                i=sample_idx,
                opacity=0.5,
                color='lightblue',
                with_contact_points=False
            )
        )

    fig = go.Figure(data=figure_data)
    fig.update_layout(
        scene=dict(aspectmode='data'),
        title=f"{title_prefix} (min={min_val:.4f}, max={max_val:.4f})",
        showlegend=True
    )
    return fig


def visualize_contact_map_parts(hand_model, object_pc, contact_value_parts,
                                part_names=None, sample_idx=0, show_hand=True,
                                marker_size=4.0, marker_opacity=0.85,
                                title_prefix='Contact Map Parts Visualization',
                                aggregate_values=None,
                                aggregate_label='整体',
                                aggregate_colorbar_title='整体 contact value'):
    """
    按照手掌及 5 根手指分别可视化 contact map
    """
    object_points = np.asarray(object_pc, dtype=np.float32)
    parts_array = np.asarray(contact_value_parts, dtype=np.float32)
    if parts_array.ndim != 2:
        raise ValueError(f"contact_value_parts 形状应为 (num_parts, N)，当前为 {parts_array.shape}")
    if object_points.shape[0] != parts_array.shape[1]:
        raise ValueError(f"object_pc 行数 ({object_points.shape[0]}) 与 contact_value_parts 列数 ({parts_array.shape[1]}) 不一致")

    num_parts = parts_array.shape[0]
    if part_names is None:
        part_names = SHADOWHAND_PART_NAMES[:num_parts]
    else:
        if len(part_names) < num_parts:
            raise ValueError("part_names 长度不足以描述所有手部部位")
        part_names = list(part_names)[:num_parts]

    traces = []
    labels = []

    def _append_trace(values, label, colorbar_title, visible):
        trace = create_contact_map_scatter(
            object_points,
            values,
            trace_name=label,
            marker_size=marker_size,
            opacity=marker_opacity,
            showscale=True,
            colorbar_title=colorbar_title,
            visible=visible
        )
        traces.append(trace)
        labels.append(label)

    if aggregate_values is not None:
        aggregate_array = np.asarray(aggregate_values, dtype=np.float32).reshape(-1)
        if aggregate_array.shape[0] != object_points.shape[0]:
            raise ValueError(f"aggregate_values 长度 ({aggregate_array.shape[0]}) 与 object_pc 行数 ({object_points.shape[0]}) 不一致")
        _append_trace(
            aggregate_array,
            aggregate_label,
            aggregate_colorbar_title,
            visible=True
        )

    for part_idx in range(num_parts):
        _append_trace(
            parts_array[part_idx],
            f"{part_names[part_idx]}",
            f"{part_names[part_idx]} contact value",
            visible=(aggregate_values is None and part_idx == 0)
        )

    hand_traces = []
    if show_hand:
        hand_traces = hand_model.get_plotly_data(
            i=sample_idx,
            opacity=0.5,
            color='lightblue',
            with_contact_points=False
        )

    fig = go.Figure(data=traces + hand_traces)

    base_visible = [
        (trace.visible if trace.visible is not None else False) if idx < len(traces) else True
        for idx, trace in enumerate(fig.data)
    ]

    buttons = []
    for idx_label, label in enumerate(labels):
        visible_state = base_visible.copy()
        for idx in range(len(traces)):
            visible_state[idx] = (idx == idx_label)
        buttons.append(dict(
            label=label,
            method='update',
            args=[
                {'visible': visible_state},
                {'title': f"{title_prefix} - {label}"}
            ]
        ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        title=f"{title_prefix} - {labels[0]}",
        showlegend=False,
        updatemenus=[
            dict(
                type='buttons',
                direction='down',
                showactive=True,
                x=1.05,
                y=0.8,
                buttons=buttons
            )
        ]
    )

    return fig


def create_contact_map_scatter(object_points, contact_values, trace_name,
                               marker_size=3.0, opacity=0.5, showscale=True,
                               colorbar_title='Contact value', visible=True):
    """
    构造带有统一色带与刻度设置的 contact map 3D 散点图
    """
    pts_array = np.asarray(object_points, dtype=np.float32)
    values_array = np.asarray(contact_values, dtype=np.float32).reshape(-1)
    if pts_array.shape[0] != values_array.shape[0]:
        raise ValueError(f"object_points 行数 ({pts_array.shape[0]}) 与 contact_values 长度 "
                         f"({values_array.shape[0]}) 不一致")

    normalized_values, min_val, max_val = _normalize_contact_values(values_array)
    tick_positions = np.linspace(0.0, 1.0, 5)
    if max_val - min_val < 1e-8:
        tick_texts = [f"{min_val:.3f}"] * len(tick_positions)
    else:
        tick_texts = [f"{min_val + (max_val - min_val) * t:.3f}" for t in tick_positions]

    marker_dict = dict(
        size=marker_size,
        opacity=opacity,
        color=normalized_values,
        cmin=0.0,
        cmax=1.0,
        colorscale=CONTACT_MAP_COLORSCALE,
        showscale=showscale
    )
    if showscale:
        marker_dict['colorbar'] = dict(
            title=colorbar_title,
            tickvals=tick_positions,
            ticktext=tick_texts
        )

    scatter = go.Scatter3d(
        x=pts_array[:, 0],
        y=pts_array[:, 1],
        z=pts_array[:, 2],
        mode='markers',
        marker=marker_dict,
        name=f"{trace_name} (min={min_val:.4f}, max={max_val:.4f})",
        customdata=values_array[:, None],
        hovertemplate=(
            'x=%{x:.4f}<br>'
            'y=%{y:.4f}<br>'
            'z=%{z:.4f}<br>'
            'value=%{customdata[0]:.4f}'
            '<extra></extra>'
        ),
        visible=visible
    )
    return scatter


def save_contact_map_html(html_path,
                          hand_model,
                          object_model,
                          sample_idx,
                          object_pose,
                          energy_stats,
                          weight_dict,
                          include_goal=True,
                          title_prefix='Contact Map Visualization',
                          marker_size=4.0):
    """
    保存带有手部和物体姿态的 contact map 可视化 HTML。
    """
    if sample_idx < 0:
        print(f"[ContactMapHTML] sample_idx={sample_idx} 非法，跳过生成: {html_path}")
        return

    goal_entry = hand_model.get_contact_map_goal(sample_idx)
    if not goal_entry:
        print(f"[ContactMapHTML] sample_idx={sample_idx} 未找到 contact map goal，跳过: {html_path}")
        return
    current_tensor = hand_model.get_contact_map_current(sample_idx)
    if current_tensor is None:
        print(f"[ContactMapHTML] sample_idx={sample_idx} 尚未计算当前 contact map，跳过: {html_path}")
        return

    object_points = goal_entry['points']
    goal_values_tensor = goal_entry.get('values', None)
    goal_parts_tensor = goal_entry.get('parts', None)

    if object_points is None or object_points.numel() == 0:
        print(f"[ContactMapHTML] sample_idx={sample_idx} 的点云为空，跳过: {html_path}")
        return

    object_points = object_points.detach().cpu().numpy()
    current_values = current_tensor.detach().cpu().numpy()

    current_parts_tensor = hand_model.get_contact_map_current_parts(sample_idx)
    current_parts = None
    if current_parts_tensor is not None:
        current_parts = current_parts_tensor.detach().cpu().numpy()

    goal_values = None
    if include_goal and goal_values_tensor is not None:
        goal_values = goal_values_tensor.detach().cpu().numpy()

    goal_parts = None
    if include_goal and goal_parts_tensor is not None:
        goal_parts = goal_parts_tensor.detach().cpu().numpy()

    target_len = object_points.shape[0]
    lengths = [target_len, current_values.shape[0]]
    if goal_values is not None:
        lengths.append(goal_values.shape[0])
    if current_parts is not None:
        lengths.append(current_parts.shape[1])
    if goal_parts is not None:
        lengths.append(goal_parts.shape[1])

    effective_len = min(lengths)
    if effective_len <= 0:
        print(f"[ContactMapHTML] sample_idx={sample_idx} contact map 长度非法，跳过: {html_path}")
        return

    if target_len != effective_len:
        object_points = object_points[:effective_len]
    if current_values.shape[0] != effective_len:
        current_values = current_values[:effective_len]
    if goal_values is not None and goal_values.shape[0] != effective_len:
        goal_values = goal_values[:effective_len]
    if current_parts is not None and current_parts.shape[1] != effective_len:
        current_parts = current_parts[:, :effective_len]
    if goal_parts is not None and goal_parts.shape[1] != effective_len:
        goal_parts = goal_parts[:, :effective_len]

    traces = []
    trace_buttons = []

    default_title = title_prefix or 'Contact Map Visualization'

    current_trace_idx = len(traces)
    current_trace = create_contact_map_scatter(
        object_points,
        current_values,
        trace_name='当前 Contact Map',
        marker_size=marker_size,
        opacity=0.85,
        showscale=True,
        colorbar_title='Contact value',
        visible=True
    )
    traces.append(current_trace)
    trace_buttons.append({
        'label': '整体-当前',
        'indices': [current_trace_idx],
        'title': f'{default_title} - 整体-当前'
    })

    goal_trace = None
    goal_trace_idx = None
    if include_goal and goal_values is not None:
        goal_trace_idx = len(traces)
        goal_trace = create_contact_map_scatter(
            object_points,
            goal_values,
            trace_name='目标 Contact Map',
            marker_size=marker_size,
            opacity=0.85,
            showscale=True,
            colorbar_title='Contact value',
            visible=False
        )
        traces.append(goal_trace)
        trace_buttons.append({
            'label': '整体-目标',
            'indices': [goal_trace_idx],
            'title': f'{default_title} - 整体-目标'
        })

    part_names = goal_entry.get('part_names') or (
        hand_model.contact_map_current_part_names[sample_idx]
        if sample_idx < len(hand_model.contact_map_current_part_names)
        else list(SHADOWHAND_PART_NAMES)
    )
    if part_names is None or len(part_names) == 0:
        part_names = list(SHADOWHAND_PART_NAMES)

    if current_parts is not None and current_parts.ndim == 2:
        num_parts = current_parts.shape[0]
        if len(part_names) < num_parts:
            part_names = list(part_names) + [f'part_{i}' for i in range(len(part_names), num_parts)]
        part_names = list(part_names)[:num_parts]

        for part_idx in range(num_parts):
            part_values = current_parts[part_idx]
            part_trace_idx = len(traces)
            part_trace = create_contact_map_scatter(
                object_points,
                part_values,
                trace_name=f'{part_names[part_idx]} 当前 Contact Map',
                marker_size=marker_size,
                opacity=0.85,
                showscale=True,
                colorbar_title='Contact value',
                visible=False
            )
            traces.append(part_trace)
            trace_buttons.append({
                'label': f'{part_names[part_idx]}-当前',
                'indices': [part_trace_idx],
                'title': f'{default_title} - {part_names[part_idx]}-当前'
            })

            if goal_parts is not None and goal_parts.ndim == 2 and goal_parts.shape[1] == object_points.shape[0]:
                if goal_parts.shape[0] >= num_parts:
                    goal_part_values = goal_parts[part_idx]
                    part_goal_trace_idx = len(traces)
                    part_goal_trace = create_contact_map_scatter(
                        object_points,
                        goal_part_values,
                        trace_name=f'{part_names[part_idx]} 目标 Contact Map',
                        marker_size=marker_size,
                        opacity=0.85,
                        showscale=True,
                        colorbar_title='Contact value',
                        visible=False
                    )
                    traces.append(part_goal_trace)
                    trace_buttons.append({
                        'label': f'{part_names[part_idx]}-目标',
                        'indices': [part_goal_trace_idx],
                        'title': f'{default_title} - {part_names[part_idx]}-目标'
                    })

    axis_range = dict(
        x=[float(object_points[:, 0].min()), float(object_points[:, 0].max())],
        y=[float(object_points[:, 1].min()), float(object_points[:, 1].max())],
        z=[float(object_points[:, 2].min()), float(object_points[:, 2].max())],
    )

    hand_trace_indices = []
    hand_opacity_default = 0.45
    hand_opacity_hidden = 0.01
    hand_traces = hand_model.get_plotly_data(
        i=sample_idx,
        opacity=hand_opacity_default,
        color='lightblue',
        with_contact_points=False
    )
    for ht in hand_traces:
        xs = np.asarray(ht.x)
        ys = np.asarray(ht.y)
        zs = np.asarray(ht.z)
        axis_range['x'][0] = float(min(axis_range['x'][0], xs.min()))
        axis_range['x'][1] = float(max(axis_range['x'][1], xs.max()))
        axis_range['y'][0] = float(min(axis_range['y'][0], ys.min()))
        axis_range['y'][1] = float(max(axis_range['y'][1], ys.max()))
        axis_range['z'][0] = float(min(axis_range['z'][0], zs.min()))
        axis_range['z'][1] = float(max(axis_range['z'][1], zs.max()))
        hand_trace_indices.append(len(traces))
        traces.append(ht)

    fig = go.Figure(data=traces)

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(range=axis_range['x']),
            yaxis=dict(range=axis_range['y']),
            zaxis=dict(range=axis_range['z']),
        ),
        title=trace_buttons[0]['title'] if trace_buttons else default_title,
        showlegend=False,
    )

    energy_lines = []
    for key, value in energy_stats.items():
        if isinstance(value, (int, float)):
            energy_lines.append(f"{key}: {value:.6f}")
        else:
            energy_lines.append(f"{key}: {value}")
    weight_json = json.dumps(weight_dict, indent=2)
    weight_html = weight_json.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
    info_text = "<b>Energy</b><br>{}<br><br><b>Weights</b><br>{}".format(
        '<br>'.join(energy_lines),
        weight_html
    )
    fig.add_annotation(
        dict(
            text=info_text,
            x=0.02,
            y=0.98,
            xref='paper',
            yref='paper',
            align='left',
            bgcolor='rgba(0, 0, 0, 0.55)',
            borderpad=8,
            font=dict(color='white', size=12),
            showarrow=False
        )
    )

    updatemenus_config = []
    if trace_buttons:
        total_traces = len(traces)
        buttons = []
        for config in trace_buttons:
            visible = [False] * total_traces
            for idx in config['indices']:
                if 0 <= idx < total_traces:
                    visible[idx] = True
            for idx in hand_trace_indices:
                if 0 <= idx < total_traces:
                    visible[idx] = True
            buttons.append(dict(
                label=config['label'],
                method='update',
                args=[
                    {'visible': visible},
                    {'title': config['title']}
                ]
            ))
        updatemenus_config.append(
            dict(
                type='buttons',
                direction='down',
                showactive=True,
                x=1.02,
                y=0.85,
                buttons=buttons
            )
        )

    if hand_trace_indices:
        opacity_arrays_default = [hand_opacity_default] * len(hand_trace_indices)
        opacity_arrays_hidden = [hand_opacity_hidden] * len(hand_trace_indices)
        hand_buttons = [
            dict(
                label='显示手部',
                method='restyle',
                args=[{'opacity': opacity_arrays_default}, hand_trace_indices]
            ),
            dict(
                label='隐藏手部',
                method='restyle',
                args=[{'opacity': opacity_arrays_hidden}, hand_trace_indices]
            )
        ]
        updatemenus_config.append(
            dict(
                type='buttons',
                direction='down',
                showactive=True,
                x=1.02,
                y=0.7,
                buttons=hand_buttons
            )
        )

    if updatemenus_config:
        fig.update_layout(updatemenus=updatemenus_config)

    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"[ContactMapHTML] 已保存: {html_path}")


def quat_to_rot6d_tensor(quat_wxyz, device):
    """
    将四元数 (w, x, y, z) 转换为 6D 旋转表示
    """
    rot_mat = transforms3d.quaternions.quat2mat(quat_wxyz)
    rot_tensor = torch.tensor(rot_mat, dtype=torch.float32, device=device)
    return torch.cat([rot_tensor[:, 0], rot_tensor[:, 1]], dim=0)

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
parser.add_argument('--run_optimization', action='store_true', help='启用能量优化流程')
parser.add_argument('--optimizer', default='annealing', type=str, choices=['annealing', 'adam'],
                    help='选择使用的优化器（annealing 或 adam）')
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
parser.add_argument('--w_cmap', default=50.0, type=float, help='weight for contact map loss')
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
parser.add_argument('--save_trajectory', action='store_true', default=False,
                    help='Enable saving intermediate optimization states for visualization')
parser.add_argument('--trajectory_interval', default=100, type=int,
                    help='Interval (in iterations) between saved trajectory snapshots')
parser.add_argument('--trajectory_sample_idx', default=[0], type=int, nargs='+',
                    help='Sample indices (within the batch) used when generating trajectory visualization')
parser.add_argument('--debug_cmap', action='store_true', default=False,
                    help='在 contact map 计算时进入调试模式并使用 Open3D 可视化当前/目标 contact map')
parser.add_argument('--cmap_html_interval', default=0, type=int,
                    help='当值大于 0 时，每隔指定迭代步保存当前 contact map 的 Plotly HTML')
parser.add_argument('--cmap_html_sample_idx', default=[0], type=int, nargs='+',
                    help='生成 contact map HTML 时需要可视化的 batch 索引列表')
parser.add_argument('--cmap_html_include_goal', action='store_true', default=False,
                    help='在 HTML 中添加目标 contact map 以便切换查看')

args = parser.parse_args()

if not isinstance(args.trajectory_sample_idx, list):
    args.trajectory_sample_idx = [args.trajectory_sample_idx]
if len(args.trajectory_sample_idx) == 0:
    args.trajectory_sample_idx = [0]
if any(idx < 0 for idx in args.trajectory_sample_idx):
    parser.error('--trajectory_sample_idx must be non-negative integers')
args.trajectory_sample_idx = list(dict.fromkeys(args.trajectory_sample_idx))
if not isinstance(args.cmap_html_sample_idx, list):
    args.cmap_html_sample_idx = [args.cmap_html_sample_idx]
if len(args.cmap_html_sample_idx) == 0:
    args.cmap_html_sample_idx = [0]
if any(idx < 0 for idx in args.cmap_html_sample_idx):
    parser.error('--cmap_html_sample_idx must be non-negative integers')
args.cmap_html_sample_idx = list(dict.fromkeys(args.cmap_html_sample_idx))
if args.cmap_html_interval < 0:
    parser.error('--cmap_html_interval must be >= 0')

if args.save_trajectory:
    if args.trajectory_interval <= 0:
        parser.error('--trajectory_interval must be a positive integer')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)

# TODO： modify the grasp file name
grasp_file = "bodex_1obj_bottle_good" # bodex_1obj_bottle, dexonomy_1obj_fin_cmap
result_path = "/home/guizhewei/guizhewei/grasp_pose_dataset/unoptimized/bodex" # /home/guizhewei/guizhewei/grasp_pose_dataset/unoptimized/bodex ,/home/guizhewei/guizhewei/grasp_pose_dataset/unoptimized
# 加载所有数据（去掉 [:10] 限制）
data_dict = np.load(os.path.join(result_path, grasp_file + '.npy'), allow_pickle=True)
print(f"成功加载 {len(data_dict)} 个抓取姿态")

object_code_list = []
hand_pose_list = []
shadow_hand_pose_list = []
obj_idx = []
object_pose_list = []
scene_scale_list = []
obj_scale_list = []
contact_map_list = []  # 存储 contact map 数据
skipped_count = 0

# TODO: debug first 10
data_dict = data_dict[50:100]

for i, data in enumerate(data_dict):
    if args.object_code is not None and data['object_code'] != args.object_code:
        continue
    qpos = data['qpos']
    object_code_list.append(data['object_code'])
    obj_idx.append(data['idx'])
    object_pose = data.get('object_pose', None)
    if object_pose is None:
        object_pose = np.eye(4, dtype=np.float32)
    else:
        object_pose = np.array(object_pose, dtype=np.float32)
        if object_pose.shape != (4, 4):
            object_pose = np.eye(4, dtype=np.float32)
    object_pose_list.append(object_pose)
    rot = data['hand_rot6d']
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
    hand_pose_list.append(torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device))

    shadow_qpos = data.get('shadow_qpos', None)
    if shadow_qpos is not None:
        shadow_qpos = np.asarray(shadow_qpos, dtype=np.float32)
        expected_len = 7 + len(shadow_joint_names)
        if shadow_qpos.shape[0] != expected_len:
            print(f"  警告: shadow_qpos 长度为 {shadow_qpos.shape[0]}, 期望 {expected_len}, 跳过该条目可视化")
            shadow_hand_pose_list.append(None)
        else:
            shadow_pose_vec = torch.zeros(3 + 6 + len(shadow_joint_names), dtype=torch.float32, device=device)
            shadow_pose_vec[:3] = torch.tensor(shadow_qpos[:3], dtype=torch.float32, device=device)
            quat = shadow_qpos[3:7]
            shadow_pose_vec[3:9] = quat_to_rot6d_tensor(quat, device)
            shadow_pose_vec[9:] = torch.tensor(shadow_qpos[7:], dtype=torch.float32, device=device)
            shadow_hand_pose_list.append(shadow_pose_vec)
    else:
        print("  警告: 数据中缺少 shadow_qpos, 跳过该条目可视化")
        shadow_hand_pose_list.append(None)
    
    # 读取 scene_scale 和 obj_scale
    scene_scale_raw = data.get('scene_scale', None)
    if scene_scale_raw is None:
        scene_scale_scalar = 1.0
    else:
        if isinstance(scene_scale_raw, (list, tuple, np.ndarray)):
            scene_scale_scalar = float(np.asarray(scene_scale_raw, dtype=np.float32).reshape(-1)[0])
        else:
            scene_scale_scalar = float(scene_scale_raw)
    scene_scale_list.append(scene_scale_raw)
    
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
        if 'contact_map_object_parts' in data:
            contact_map_data['contact_value_parts'] = data['contact_map_object_parts']
        if 'contact_map_object_part_names' in data:
            contact_map_data['contact_value_part_names'] = data['contact_map_object_part_names']
        # print(f"  加载 contact map: obj_points={contact_map_data['object_point_cloud'].shape}, "
        #       f"contact_value range=[{contact_map_data['contact_value'].min():.3f}, {contact_map_data['contact_value'].max():.3f}]")
    else:
        print(f"  警告: 数据中缺少 contact map 信息")
    contact_map_list.append(contact_map_data)
    if args.object_code is not None:
        break

hand_pose_tensor = torch.stack([hp.view(-1) for hp in hand_pose_list], dim=0).to(device)

shadow_hand_available_mask = [pose is not None for pose in shadow_hand_pose_list]
shadow_hand_pose_tensor = None
if any(shadow_hand_available_mask):
    filled_shadow_poses = [
        pose if pose is not None else torch.zeros(3 + 6 + len(shadow_joint_names), dtype=torch.float32, device=device)
        for pose in shadow_hand_pose_list
    ]
    shadow_hand_pose_tensor = torch.stack(filled_shadow_poses, dim=0)
else:
    if len(shadow_hand_pose_list) > 0:
        print("警告: 所有样本均缺少 shadow_qpos，无法生成 Shadow Hand 可视化")

total_batch_size = len(object_code_list) * args.batch_size

trajectory_records = []
trajectory_context = {
    'sample_indices': [],
    'object_pc': None,
    'contact_goal': None,
    'contact_goal_parts': None,
}
if args.save_trajectory:
    if total_batch_size == 0:
        print("Warning: no samples available for trajectory recording, disabling --save_trajectory")
        args.save_trajectory = False
    else:
        max_idx = total_batch_size - 1
        adjusted_indices = []
        for idx in args.trajectory_sample_idx:
            if idx >= total_batch_size:
                print(f"trajectory_sample_idx {idx} is out of range, using {max_idx} instead")
                idx = max_idx
            adjusted_indices.append(idx)
        trajectory_context['sample_indices'] = sorted(set(adjusted_indices))
        if not trajectory_context['sample_indices']:
            trajectory_context['sample_indices'] = [0]
        args.trajectory_sample_idx = trajectory_context['sample_indices']
# 处理 contact map HTML 采样索引
cmap_html_sample_indices = []
if args.cmap_html_interval > 0:
    if total_batch_size == 0:
        print("Warning: no samples available for contact map HTML export, disabling --cmap_html_interval")
        args.cmap_html_interval = 0
    else:
        max_idx = total_batch_size - 1
        adjusted_cmap_indices = []
        for idx in args.cmap_html_sample_idx:
            if idx >= total_batch_size:
                print(f"cmap_html_sample_idx {idx} is out of range, using {max_idx} instead")
                idx = max_idx
            adjusted_cmap_indices.append(idx)
        cmap_html_sample_indices = sorted(set(adjusted_cmap_indices))
        if not cmap_html_sample_indices:
            cmap_html_sample_indices = [0]
        args.cmap_html_sample_idx = cmap_html_sample_indices
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

if args.debug_cmap:
    hand_model.set_cmap_debug(True)

# Change back to original directory
os.chdir(current_dir)

object_model = ObjectModel(
    data_root_path='/home/guizhewei/guizhewei/DexGraspNet/data/1_obj',
    batch_size_each=args.batch_size,
    device=device
)
object_model.initialize(object_code_list=object_code_list, scene_scale_list=scene_scale_list, obj_scale_list=obj_scale_list)

# 打印 scale 信息用于调试
# print("\n=== Scale 信息 ===")
# for i in range(len(object_code_list)):
#     print(f"Object {i}: {object_code_list[i]}")
#     print(f"  scene_scale: {scene_scale_list[i]}")
#     print(f"  obj_scale: {obj_scale_list[i]}")
#     if scene_scale_list[i] is not None and obj_scale_list[i] is not None:
#         if isinstance(obj_scale_list[i], np.ndarray):
#             obj_scale_val = np.mean(obj_scale_list[i]) if obj_scale_list[i].ndim == 1 else obj_scale_list[i].flat[0]
#         else:
#             obj_scale_val = obj_scale_list[i]
#         total_scale = float(scene_scale_list[i]) * float(obj_scale_val)
#         print(f"  total_scale (scene * obj): {total_scale}")
#     print(f"  object_scale_tensor: {object_model.object_scale_tensor[i][0].item()}")
# print("=" * 20 + "\n")

hand_st_plotly = []
# contact_point_indices = torch.randint(hand_model.n_contact_candidates, size=[total_batch_size, args.n_contact], device=device)
contact_point_indices = torch.arange(
    hand_model.n_contact_candidates, device=device
).repeat(total_batch_size, 1)
hand_pose_tensor.requires_grad_()
# import ipdb; ipdb.set_trace()
hand_model.set_parameters(hand_pose_tensor, contact_point_indices)

shadow_hand_model = None
if shadow_hand_pose_tensor is not None:
    shadow_hand_model = HandModel(
        mjcf_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/shadow_hand_wrist_free.xml',
        mesh_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/meshes_shadow',
        contact_points_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/contact_points.json',
        penetration_points_path='/home/guizhewei/guizhewei/DexGraspNet/grasp_generation/mjcf/penetration_points.json',
        device=device
    )
    shadow_hand_model.set_parameters(shadow_hand_pose_tensor)

hand_model.allocate_contact_map_goals(total_batch_size)

# 设置 contact map goal（如果数据中包含）
print("\n=== 设置 Contact Map Goal ===")
for i, contact_map_data in enumerate(contact_map_list):
    sample_base_idx = i * args.batch_size
    if len(contact_map_data) == 0:
        print(f"Object {i}: 未找到 contact map 数据，跳过")
        continue

    obj_pc = contact_map_data.get('object_point_cloud', None)
    obj_normal = contact_map_data.get('object_normal_cloud', None)
    contact_val = contact_map_data.get('contact_value', None)
    if obj_pc is None or contact_val is None:
        print(f"Object {i}: contact map 数据不完整，跳过")
        continue

    goal_dict = {
        'object_point_cloud': obj_pc,
        'object_normal_cloud': obj_normal,
        'contact_value': contact_val,
        'contact_value_part_names': contact_map_data.get('contact_value_part_names', SHADOWHAND_PART_NAMES),
    }
    goal_parts_np = contact_map_data.get('contact_value_parts', None)
    part_names = goal_dict['contact_value_part_names']

    for j in range(args.batch_size):
        sample_idx_global = sample_base_idx + j
        if sample_idx_global >= total_batch_size:
            break
        hand_model.set_contact_map_goal(
            goal_dict,
            contact_map_goal_parts=goal_parts_np,
            sample_idx=sample_idx_global,
            part_names=part_names
        )

    try:
        part_names_list = list(part_names)
    except Exception:
        part_names_list = list(SHADOWHAND_PART_NAMES)

    obj_pc_np = np.asarray(obj_pc, dtype=np.float32)
    contact_val_np = np.asarray(contact_val, dtype=np.float32).reshape(-1)
    if goal_parts_np is not None and isinstance(goal_parts_np, np.ndarray) and goal_parts_np.size > 0:
        goal_fig = visualize_contact_map_parts(
            hand_model,
            obj_pc_np,
            np.asarray(goal_parts_np, dtype=np.float32),
            part_names=part_names_list,
            sample_idx=sample_base_idx,
            show_hand=False,
            title_prefix=f'Contact Map Visualization - Object {i}',
            aggregate_values=contact_val_np,
            aggregate_label='整体',
            aggregate_colorbar_title='整体 contact value'
        )
    else:
        goal_fig = visualize_contact_map_with_custom_colors(
            hand_model,
            obj_pc_np,
            contact_val_np,
            sample_idx=sample_base_idx,
            show_hand=False,
            title_prefix=f'Contact Map Visualization - Object {i}'
        )
    # goal_fig.show()
    print(f"Object {i}: 设置 contact map goal, 点数={obj_pc_np.shape[0]}")
    print(f"  Contact value stats: mean={contact_val_np.mean():.4f}, "
          f"max={contact_val_np.max():.4f}, min={contact_val_np.min():.4f}")

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
optimizer = None
if args.run_optimization:
    optimizer_map = {
        'annealing': Annealing,
        'adam': Adam,
    }
    optimizer_cls = optimizer_map[args.optimizer]
    print(f"使用优化器: {args.optimizer}")
    optimizer = optimizer_cls(hand_model, init_hand_pose=hand_pose_tensor, **optim_config)

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

# Set logger in hand_model for tensorboard logging
hand_model.logger = logger

# contact map HTML 输出目录
cmap_html_output_dir = None
if args.cmap_html_interval > 0:
    cmap_html_output_dir = os.path.join('../data/experiments', args.name, 'cmap_html')
    os.makedirs(cmap_html_output_dir, exist_ok=True)

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

def maybe_export_cmap_html(step: int,
                           energy_components=None,
                           refresh_contact_map: bool = False,
                           force: bool = False):
    """
    根据配置在指定迭代步导出 contact map HTML，可选地刷新能量与 contact map。

    Parameters
    ----------
    step: int
        当前迭代步（0 表示初始状态）
    energy_components: Optional[Tuple[torch.Tensor, ...]]
        已计算好的能量分量 (energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_cmap)
    refresh_contact_map: bool
        若为 True，会在导出前重新计算能量，确保 contact map 与当前姿态一致
    force: bool
        若为 True，无视 step 与 interval 的关系直接导出
    """
    if args.cmap_html_interval <= 0 or cmap_html_output_dir is None:
        return energy_components

    if not force:
        if step != 0 and (step % args.cmap_html_interval) != 0:
            return energy_components

    components = energy_components
    if components is None or refresh_contact_map:
        with torch.no_grad():
            components = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    if components is None:
        return energy_components

    energy_t, E_fc_t, E_dis_t, E_pen_t, E_spen_t, E_joints_t, E_cmap_t = components
    stats = {
        'step': int(step),
        'E': float(energy_t.mean().item()),
        'E_fc': float(E_fc_t.mean().item()),
        'E_dis': float(E_dis_t.mean().item()),
        'E_pen': float(E_pen_t.mean().item()),
        'E_spen': float(E_spen_t.mean().item()),
        'E_joints': float(E_joints_t.mean().item()),
        'E_cmap': float(E_cmap_t.mean().item()),
    }

    for sample_idx in args.cmap_html_sample_idx:
        sample_idx = int(sample_idx)
        if sample_idx < 0 or sample_idx >= total_batch_size:
            print(f"[ContactMapHTML] sample_idx {sample_idx} 超出范围，跳过")
            continue

        if object_pose_list:
            object_list_idx = min(len(object_pose_list) - 1, sample_idx // args.batch_size)
            object_pose = object_pose_list[object_list_idx]
        else:
            object_pose = None

        html_path = os.path.join(
            cmap_html_output_dir,
            f'step_{step:06d}_sample_{sample_idx}.html'
        )
        save_contact_map_html(
            html_path=html_path,
            hand_model=hand_model,
            object_model=object_model,
            sample_idx=sample_idx,
            object_pose=object_pose,
            energy_stats=stats,
            weight_dict=weight_dict,
            include_goal=args.cmap_html_include_goal,
            title_prefix=f'Contact Map Step {step} (Sample {sample_idx})'
        )

    return components

if args.save_trajectory:
    def record_trajectory_snapshot(step_label: int):
        with torch.no_grad():
            snap_energy, snap_E_fc, snap_E_dis, snap_E_pen, snap_E_spen, snap_E_joints, snap_E_cmap = cal_energy(
                hand_model, object_model, verbose=True, **weight_dict
            )
        record = {
            'step': int(step_label),
            'hand_pose': hand_model.hand_pose.detach().cpu().numpy().copy(),
            'contact_point_indices': hand_model.contact_point_indices.detach().cpu().numpy().copy(),
            'energy': snap_energy.detach().cpu().numpy().copy(),
            'E_fc': snap_E_fc.detach().cpu().numpy().copy(),
            'E_dis': snap_E_dis.detach().cpu().numpy().copy(),
            'E_pen': snap_E_pen.detach().cpu().numpy().copy(),
            'E_spen': snap_E_spen.detach().cpu().numpy().copy(),
            'E_joints': snap_E_joints.detach().cpu().numpy().copy(),
            'E_cmap': snap_E_cmap.detach().cpu().numpy().copy(),
            'contact_map': None,
            'contact_map_parts': None,
        }
        if hand_model.contact_value_current_per_sample:
            record['contact_map'] = [
                cv.detach().cpu().numpy().copy() if isinstance(cv, torch.Tensor) else None
                for cv in hand_model.contact_value_current_per_sample
            ]
        if hand_model.contact_value_current_parts_per_sample:
            record['contact_map_parts'] = [
                cp.detach().cpu().numpy().copy() if isinstance(cp, torch.Tensor) else None
                for cp in hand_model.contact_value_current_parts_per_sample
            ]
        if trajectory_context['object_pc'] is None and hand_model.object_point_cloud is not None:
            trajectory_context['object_pc'] = hand_model.object_point_cloud.detach().cpu().numpy().copy()
        if trajectory_context['contact_goal'] is None and hand_model.contact_value_goal is not None:
            trajectory_context['contact_goal'] = hand_model.contact_value_goal.detach().cpu().numpy().copy()
        if trajectory_context['contact_goal_parts'] is None and hand_model.contact_value_goal_parts is not None:
            trajectory_context['contact_goal_parts'] = hand_model.contact_value_goal_parts.detach().cpu().numpy().copy()
        trajectory_records.append(record)
energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_cmap = cal_energy(hand_model, object_model, verbose=True, **weight_dict)
print('Initial energy:', energy.mean().item(),
      ' E_fc:', E_fc.mean().item(),
      ' E_dis:', E_dis.mean().item(),
      ' E_pen:', E_pen.mean().item(),   
    ' E_spen:', E_spen.mean().item(),
        ' E_joints:', E_joints.mean().item(),
        ' E_cmap:', E_cmap.mean().item())

energy.sum().backward(retain_graph=True)
hand_model.current_step = 0  # Set initial step for logging
logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, 0, E_cmap=E_cmap, show=True)

energy_components = maybe_export_cmap_html(
    step=0,
    energy_components=(energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_cmap),
    refresh_contact_map=False,
    force=True
)
energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_cmap = energy_components

if args.run_optimization and optimizer is not None:
    pbar = tqdm(range(1, args.n_iter + 1), desc='optimizing', dynamic_ncols=True)

    for step in pbar:
        hand_model.current_step = step

        _ = optimizer.try_step(fix_wrist=args.fix_wrist)

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints, new_E_cmap = cal_energy(
            hand_model, object_model, verbose=True, **weight_dict
        )

        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, _ = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_fc[accept] = new_E_fc[accept]
            E_dis[accept] = new_E_dis[accept]
            E_pen[accept] = new_E_pen[accept]
            E_spen[accept] = new_E_spen[accept]
            E_joints[accept] = new_E_joints[accept]
            E_cmap[accept] = new_E_cmap[accept]

            logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, step, E_cmap=E_cmap, show=False)

            if args.save_trajectory and (step % args.trajectory_interval == 0):
                record_trajectory_snapshot(step)

        energy_components = maybe_export_cmap_html(
            step=step,
            energy_components=(energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_cmap),
            refresh_contact_map=True
        )
        energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_cmap = energy_components

        pbar.set_postfix({
            "E": f"{energy.mean().item():.3f}",
            "fc": f"{E_fc.mean().item():.3f}",
            "dis": f"{E_dis.mean().item():.3f}",
            "pen": f"{E_pen.mean().item():.3f}",
            "spen": f"{E_spen.mean().item():.3f}",
            "joints": f"{E_joints.mean().item():.3f}",
            "cmap": f"{E_cmap.mean().item():.3f}"
        })

    if args.cmap_html_interval > 0 and (args.n_iter % args.cmap_html_interval) != 0:
        energy_components = maybe_export_cmap_html(
            step=args.n_iter,
            energy_components=(energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_cmap),
            refresh_contact_map=True,
            force=True
        )
        energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_cmap = energy_components

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