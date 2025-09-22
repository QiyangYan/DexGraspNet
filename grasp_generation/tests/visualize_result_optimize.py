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
    "R_thumb_MCP_joint1",
    "R_thumb_MCP_joint2",
    "R_thumb_PIP_joint",
    "R_thumb_DIP_joint",
    "R_index_MCP_joint",
    "R_index_DIP_joint",
    "R_middle_MCP_joint",
    "R_middle_DIP_joint",
    "R_ring_MCP_joint",
    "R_ring_DIP_joint",
    "R_pinky_MCP_joint",
    "R_pinky_DIP_joint"
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code', type=str, default='46_008_pudding_box')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='/home/ubuntu/Documents/DexGraspNet/data/experiments/exp_2/results_d1_v0')
    args = parser.parse_args()

    device = 'cpu'

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

    hand_model = HandModel(
    mjcf_path='/home/ubuntu/Documents/DexGraspNet/grasp_generation/mjcf/inspire_free_dexgraspnet.xml',
    mesh_path='/home/ubuntu/Documents/DexGraspNet/grasp_generation/mjcf/meshes',
    contact_points_path='/home/ubuntu/Documents/DexGraspNet/grasp_generation/mjcf/contact_points_inspire.json',
    penetration_points_path='/home/ubuntu/Documents/DexGraspNet/grasp_generation/mjcf/penetration_points_inspire.json',
    device=device
    )

    object_model = ObjectModel(
        data_root_path='/home/ubuntu/Documents/DexYCB/models',
        batch_size_each=1,
        num_samples=2000, 
        device=device
    )
    name_no_ext = os.path.splitext(args.object_code)[0]
    object_idx, object_code = name_no_ext.split("_", 1)
    object_model.initialize(object_code_list=object_code)
    object_model.object_scale_tensor = torch.tensor(1, dtype=torch.float, device=device).reshape(1, 1)

    # visualize

    hand_st_plotly = []
    hand_model.set_parameters(hand_pose.unsqueeze(0))
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
    if 'energy' in data_dict:
        energy = data_dict['energy']
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        # E_spen = round(data_dict['E_spen'], 5)
        E_joints = round(data_dict['E_joints'], 5)
        result = f'Index {args.num}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}'
        fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
    fig.update_layout(scene_aspectmode='data')
    fig.show()
