"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: energy functions
"""

import torch
import torch.nn.functional as F

def cal_energy(hand_model, object_model, w_dis=100.0, w_pen=100.0, w_spen=10.0, w_joints=1.0, w_cmap=1.0, cmap_energy_func='align_dist', verbose=False):
    batch_size, n_contact, _ = hand_model.contact_points.shape
    E_fc = torch.zeros(batch_size, dtype=torch.float, device=hand_model.device)
    E_dis = torch.zeros(batch_size, dtype=torch.float, device=hand_model.device)
    E_pen = torch.zeros(batch_size, dtype=torch.float, device=hand_model.device)
    E_joints = torch.zeros(batch_size, dtype=torch.float, device=hand_model.device)
    E_spen = torch.zeros(batch_size, dtype=torch.float, device=hand_model.device)

    # E_dis
    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = object_model.device
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    # E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)
    
    # E_pen_dexgraspnet
    # object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    # object_surface_points = object_model.surface_points_tensor * object_scale  # (n_objects * batch_size_each, num_samples, 3)
    # distances = hand_model.cal_distance(object_surface_points)
    # # hand_model.viz_cal_distance(object_surface_points, batch_idx=0, mode="max")
    # distances[distances <= 0] = 0
    # E_pen = distances.sum(-1)
    
    # E_pen by obj surface points
    hand_surface_points = hand_model.get_surface_points()
    surface_distance, contact_normal = object_model.cal_distance(hand_surface_points)
    E_pen = F.relu(surface_distance).sum(dim=-1) * 20 # 50 is pretty good for maintaining the posture

    # Visualize hand_surface_points using plotly if verbose is True
    # if verbose:
    #     try:
    #         import plotly.graph_objects as go
    #         import numpy as np
    #         # Convert to cpu and numpy for plotting
    #         pts = hand_surface_points[0].detach().cpu().numpy() if isinstance(hand_surface_points, torch.Tensor) else hand_surface_points
    #         fig = go.Figure(
    #             data=[go.Scatter3d(
    #                 x=pts[:,0], y=pts[:,1], z=pts[:,2],
    #                 mode='markers',
    #                 marker=dict(size=3, color='blue'),
    #                 name='Hand Surface Points'
    #             )]
    #         )
    #         fig.update_layout(title="Hand Surface Points", scene_aspectmode='data')
    #         fig.show()
    #     except Exception as e:
    #         print(f"[Hand Surface Points Visualization failed]: {e}")


    
    # import pdb; pdb.set_trace()
    # E_fc

    # contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)
    # transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
    #                                       [0, 0, 1, 0, 0, 0, -1, 0, 0],
    #                                       [0, -1, 0, 1, 0, 0, 0, 0, 0]],
    #                                      dtype=torch.float, device=device)
    # g = torch.cat([torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3),
    #                (hand_model.contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)], 
    #               dim=2).float().to(device)
    # norm = torch.norm(contact_normal @ g, dim=[1, 2])
    # E_fc = norm * norm

    # E_joints
    # E_joints = torch.sum((hand_model.hand_pose[:, 9:] > hand_model.joints_upper) * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper), dim=-1) + \
    #     torch.sum((hand_model.hand_pose[:, 9:] < hand_model.joints_lower) * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]), dim=-1)
    
    
    # E_pen
    # object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    # object_surface_points = object_model.surface_points_tensor * object_scale  # (n_objects * batch_size_each, num_samples, 3)
    # distances = hand_model.cal_distance(object_surface_points)
    # # hand_model.viz_cal_distance(object_surface_points, batch_idx=0, mode="max")
    # distances[distances <= 0] = 0
    # E_pen = -distances.sum(-1)
    # # d = hand_model.cal_distance(object_surface_points).detach()
    # # # print("d stats: min=", d.min().item(), "max=", d.max().item())
    # print("inside count (d>0):", (d>0).sum().item(), " outside (d<0):", (d<0).sum().item())

    # E_cm
    # E_cm = hand_model.get_cmap_loss()


    # E_spen
    E_spen = hand_model.self_penetration()

    # E_cmap (using CMapAdam-style computation)
    E_cmap = hand_model.get_cmap_loss(energy_func_name=cmap_energy_func)

    # if verbose:
    #     return E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints, E_fc, E_dis, E_pen, E_spen, E_joints
    # else:
    #     return E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints

    if verbose:
        return E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints + w_cmap * E_cmap, E_fc, E_dis, E_pen, E_spen, E_joints, E_cmap
    else:
        return E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints + w_cmap * E_cmap
