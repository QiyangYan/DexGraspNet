"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: Class HandModel
"""


import os
import json
import numpy as np
import torch
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import pytorch_kinematics as pk
import plotly.graph_objects as go
import pytorch3d.structures
import pytorch3d.ops
import trimesh as tm
from torchsdf import index_vertices_by_faces, compute_sdf


class HandModel:
    def __init__(self, mjcf_path, mesh_path, contact_points_path, penetration_points_path, n_surface_points=0, device='cpu'):
        """
        Create a Hand Model for a MJCF robot
        
        Parameters
        ----------
        mjcf_path: str
            path to mjcf file
        mesh_path: str
            path to mesh directory
        contact_points_path: str
            path to hand-selected contact candidates
        penetration_points_path: str
            path to hand-selected penetration keypoints
        n_surface_points: int
            number of points to sample from surface of hand, use fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device
        
        # load articulation
        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())
        
        # load contact points and penetration points
        
        contact_points = json.load(open(contact_points_path, 'r')) if contact_points_path is not None else None
        penetration_points = json.load(open(penetration_points_path, 'r')) if penetration_points_path is not None else None

        # build mesh

        self.mesh = {}
        areas = {}

        def build_mesh_recurse(body):
            if(len(body.link.visuals) > 0):
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
                    # Skip unsupported or collision-only geometry types
                    if visual.geom_type in ["box", "sphere", None]:
                        continue
                    elif visual.geom_type == "capsule":
                        link_mesh = tm.primitives.Capsule(radius=visual.geom_param[0], height=visual.geom_param[1] * 2).apply_translation((0, 0, -visual.geom_param[1]))
                    elif visual.geom_type == "cylinder":
                        link_mesh = tm.primitives.Cylinder(radius=visual.geom_param[0], height=visual.geom_param[1] * 2).apply_translation((0, 0, -visual.geom_param[1]))
                    elif visual.geom_type == "mesh":
                        link_mesh = tm.load_mesh(os.path.join(mesh_path, visual.geom_param[0]+".STL"), process=False)
                        if visual.geom_param[1] is not None:
                            scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)
                    else:
                        raise ValueError(f"Unknown geom type {visual.geom_type} for link {link_name}")
                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                    pos = visual.offset.to(self.device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                contact_candidates = torch.tensor(contact_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3) if contact_points is not None else None
                # penetration_keypoints = torch.tensor(penetration_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3) if penetration_points is not None else None
                self.mesh[link_name] = {
                    'vertices': link_vertices,
                    'faces': link_faces,
                    'contact_candidates': contact_candidates,
                    # 'penetration_keypoints': penetration_keypoints,
                }
                link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
                self.mesh[link_name]['face_verts'] = link_face_verts
                areas[link_name] = tm.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()
            for children in body.children:
                build_mesh_recurse(children)
        
        build_mesh_recurse(self.chain._root)

        # set joint limits

        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(body.joint.range[0])
                self.joints_upper.append(body.joint.range[1])
            for children in body.children:
                set_joint_range_recurse(children)
        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.stack(self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(device)

        # sample surface points

        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]['surface_points'] = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(self.mesh[link_name]['vertices'].unsqueeze(0), self.mesh[link_name]['faces'].unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]['surface_points'] = surface_points

        # indexing

        self.link_name_to_link_index = dict(zip([link_name for link_name in self.mesh], range(len(self.mesh))))

        self.contact_candidates = [self.mesh[link_name]['contact_candidates'] for link_name in self.mesh]
        self.global_index_to_link_index = sum([[i] * len(contact_candidates) for i, contact_candidates in enumerate(self.contact_candidates)], [])
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(self.global_index_to_link_index, dtype=torch.long, device=device)
        self.n_contact_candidates = self.contact_candidates.shape[0]

        # self.penetration_keypoints = [self.mesh[link_name]['penetration_keypoints'] for link_name in self.mesh]
        # self.global_index_to_link_index_penetration = sum([[i] * len(penetration_keypoints) for i, penetration_keypoints in enumerate(self.penetration_keypoints)], [])
        # self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        # self.global_index_to_link_index_penetration = torch.tensor(self.global_index_to_link_index_penetration, dtype=torch.long, device=device)
        # self.n_keypoints = self.penetration_keypoints.shape[0]

        # parameters

        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None

    def set_parameters(self, hand_pose, contact_point_indices=None, init=False):
        """
        Set translation, rotation, joint angles, and contact points of grasps
        
        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(self.hand_pose[:, 3:9])
        self.current_status = self.chain.forward_kinematics(self.hand_pose[:, 9:])
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device)
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, n_contact, 4, 4)
                transforms[mask] = cur[mask]
            # link_names = sorted(self.mesh.keys(), key=lambda n: self.link_name_to_link_index[n])
            # link_T = torch.stack(
            #     [self.current_status[ln].get_matrix() for ln in link_names], dim=1
            # )
            # b_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(batch_size, n_contact)
            # transforms = link_T[b_idx, link_indices]                                            
            self.contact_points = torch.cat([self.contact_points, torch.ones(batch_size, n_contact, 1, dtype=torch.float, device=self.device)], dim=2)
            self.contact_points = (transforms @ self.contact_points.unsqueeze(3))[:, :, :3, 0]
            self.contact_points = self.contact_points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)

    def cal_distance(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes
        
        Interiors are positive, exteriors are negative
        
        Use analytical method and our modified Kaolin package
        
        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """
        # Consider each link seperately: 
        #   First, transform x into each link's local reference frame using inversed fk, which gives us x_local
        #   Next, calculate point-to-mesh distances in each link's frame, this gives dis_local
        #   Finally, the maximum over all links is the final distance from one point to the entire ariticulation
        # In particular, the collision mesh of ShadowHand is only composed of Capsules and Boxes
        # We use analytical method to calculate Capsule sdf, and use our modified Kaolin package for other meshes
        # This practice speeds up the reverse penetration calculation
        # Note that we use a chamfer box instead of a primitive box to get more accurate signs
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            if link_name in ['R_thumb_proximal_base_child']:
                continue
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            if 'geom_param' not in self.mesh[link_name]:
                face_verts = self.mesh[link_name]['face_verts']
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            else:
                height = self.mesh[link_name]['geom_param'][1] * 2
                height = torch.tensor([height], dtype=torch.float)
                radius = self.mesh[link_name]['geom_param'][0]
                nearest_point = x_local.detach().clone()
                nearest_point[:, :2] = 0
                nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], 0, height)
                dis_local = radius - (x_local - nearest_point).norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    import plotly.graph_objects as go
    import torch

    def viz_cal_distance(self, x, batch_idx=0, max_points=6000, mode="max", link_name=None):
        """
        Visualize cal_distance() inputs/outputs for one batch item.

        x: (B, N, 3) world-space object points (same tensor passed to cal_distance)
        batch_idx: which batch item to show
        max_points: subsample for speed
        mode: "max" (color by max over links) or "link" (color by one link)
        link_name: when mode="link", which link to color by
        """
        assert 0 <= batch_idx < x.shape[0]
        B, N, _ = x.shape
        dev = x.device

        # ---- 1) Transform points world->hand base (same as cal_distance)
        x_hand = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation

        # ---- 2) Compute per-link signed distances (same logic as cal_distance)
        per_link_d = []
        valid_links = [ln for ln in self.mesh if ln not in ['R_thumb_proximal_base_child']]
        face_cache = {}

        for ln in valid_links:
            M = self.current_status[ln].get_matrix()  # (B,4,4)
            x_local = (x_hand - M[:, :3, 3].unsqueeze(1)) @ M[:, :3, :3]          # (B,N,3)
            x_local_flat = x_local.reshape(-1, 3)                                  # (B*N,3)

            if 'geom_param' not in self.mesh[ln]:
                if ln not in face_cache:
                    face_cache[ln] = self.mesh[ln]['face_verts']                   # (F,3,3) link-local
                face_verts = face_cache[ln]
                dis_local, dis_signs, _, _ = compute_sdf(x_local_flat, face_verts) # unsigned^2, signs
                d = torch.sqrt(dis_local + 1e-8) * (-dis_signs)                    # signed: inside +
            else:
                # Capsule analytic (same as your code)
                height = self.mesh[ln]['geom_param'][1] * 2
                height = torch.tensor([height], dtype=torch.float, device=dev)
                radius = self.mesh[ln]['geom_param'][0]
                nearest = x_local_flat.detach().clone()
                nearest[:, :2] = 0
                nearest[:, 2] = torch.clamp(nearest[:, 2], 0, height)
                d = radius - (x_local_flat - nearest).norm(dim=1)

            per_link_d.append(d.view(B, N))

        per_link_d = torch.stack(per_link_d, dim=0)            # (L,B,N)

        # ---- which colors to show on points
        if mode == "max":
            d_show, which = per_link_d.max(dim=0)              # (B,N), (B,N) argmax over links
            colors = d_show[batch_idx]                         # (N,)
            # most frequent responsible link for highlight
            link_idx = which[batch_idx].mode().values.item()
            show_link = valid_links[link_idx]
        elif mode == "link":
            assert link_name in valid_links, f"link_name must be one of {valid_links}"
            link_idx = valid_links.index(link_name)
            colors = per_link_d[link_idx][batch_idx]
            show_link = link_name
        else:
            raise ValueError("mode must be 'max' or 'link'")

        # ---- 3) Subsample points for plotting
        idx = torch.randperm(N, device=dev)[:min(N, max_points)]
        pts = x[batch_idx, idx].detach().cpu().numpy()
        cols = colors[idx].detach().cpu().numpy()

        # ---- NEW: per-link penetration flag for coloring meshes
        # A link is "penetrating" if any point has d>0 for that link.
        link_penetrates = {}
        for k, ln in enumerate(valid_links):
            link_penetrates[ln] = (per_link_d[k, batch_idx] > 0).any().item()

        # ---- 4) Build ALL hand link meshes in WORLD frame to overlay
        traces = []
        for ln in self.mesh:
            V = self.mesh[ln]['vertices']
            F = self.mesh[ln]['faces']
            if V.numel() == 0:
                continue

            M = self.current_status[ln].get_matrix()[batch_idx]       # (4,4), link in hand
            V_hand = V @ M[:3, :3].T + M[:3, 3]                       # link-local -> hand
            Rg = self.global_rotation[batch_idx]
            tg = self.global_translation[batch_idx]
            V_world = V_hand @ Rg.T + tg                               # hand -> world

            Vw = V_world.detach().cpu().numpy()
            Fw = F.detach().cpu().numpy().astype(int)

            # Color logic:
            # - For links we computed distances for: red if penetration, else green.
            # - For excluded links (e.g., bases): light gray.
            if ln in link_penetrates:
                is_pen = link_penetrates[ln]
                base_color = 'red' if is_pen else 'green'
                base_opacity = 0.45 if ln == show_link else 0.25
            else:
                base_color = 'lightgray'
                base_opacity = 0.2

            traces.append(go.Mesh3d(
                x=Vw[:, 0], y=Vw[:, 1], z=Vw[:, 2],
                i=Fw[:, 0], j=Fw[:, 1], k=Fw[:, 2],
                opacity=base_opacity,
                color=base_color,
                name=(ln + (" (penetration)" if ln in link_penetrates and link_penetrates[ln] else "")),
                showscale=False
            ))

        # ---- 5) Add point cloud colored by signed distance
        pc = go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=3, color=cols, colorscale='Turbo', showscale=True,
                        colorbar=dict(title='signed dist (+ inside)')),
            name='object pts'
        )

        fig = go.Figure(traces + [pc])
        fig.update_layout(
            title=f"cal_distance debug — batch={batch_idx}, mode={mode}, link={show_link}",
            scene_aspectmode='data',
            legend=dict(itemsizing='constant')
        )
        fig.show()

    def self_penetration(self):
        """
        Calculate self penetration energy
        
        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        batch_size = self.global_translation.shape[0]
        points = self.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        link_indices = self.global_index_to_link_index_penetration.clone().repeat(batch_size,1)
        transforms = torch.zeros(batch_size, self.n_keypoints, 4, 4, dtype=torch.float, device=self.device)
        for link_name in self.mesh:
            mask = link_indices == self.link_name_to_link_index[link_name]
            cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, self.n_keypoints, 4, 4)
            transforms[mask] = cur[mask]
        points = torch.cat([points, torch.ones(batch_size, self.n_keypoints, 1, dtype=torch.float, device=self.device)], dim=2)
        points = (transforms @ points.unsqueeze(3))[:, :, :3, 0]
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        dis = 0.02 - dis
        E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
        return E_spen.sum((1,2))

    def get_surface_points(self):
        """
        Get surface points
        
        Returns
        -------
        points: (N, `n_surface_points`, 3)
            surface points
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['surface_points'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['surface_points']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_contact_candidates(self):
        """
        Get all contact candidates
        
        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['contact_candidates'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['contact_candidates']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points
    
    def get_penetraion_keypoints(self):
        """
        Get penetration keypoints
        
        Returns
        -------
        points: (N, `n_keypoints`, 3) torch.Tensor
            penetration keypoints
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['penetration_keypoints'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['penetration_keypoints']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_plotly_data(self, i, opacity=0.5, color='lightblue', with_contact_points=False, pose=None):
        """
        Get visualization data for plotly.graph_objects
        
        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix
        
        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            data.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color, opacity=opacity))
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            data.append(go.Scatter3d(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2], mode='markers', marker=dict(color='red', size=5)))
        return data
    
    def get_cmap_loss(self, prediction, target):
        #  -> Dict[str, Tensor]
        # chamfer loss between predict-hand point cloud and target-hand point cloud
        pred_hand_pc = self.get_surface_points()
        target_hand_pc = self.target_hand_pc
        pc = target["matched"]["obj_pc"]
        pred_cmap = contact_map_of_m_to_n(pc, pred_hand_pc)
        gt_cmap = contact_map_of_m_to_n(pc, target_hand_pc)
        cmap_loss = torch.nn.functional.mse_loss(pred_cmap, gt_cmap, reduction='mean')
        loss = {"cmap": cmap_loss}
        return loss

def min_distance_from_m_to_n(m, n):
    """
    :param m: [..., M, 3]
    :param n: [..., N, 3]
    :return: [..., M]
    """
    m_num = m.shape[-2]
    n_num = n.shape[-2]

    # m_: [..., M, N, 3]
    # n_: [..., M, N, 3]
    m_ = m.unsqueeze(-2)  # [..., M, 1, 3]
    n_ = n.unsqueeze(-3)  # [..., 1, N, 3]

    m_ = torch.repeat_interleave(m_, n_num, dim=-2)  # [..., M, N, 3]
    n_ = torch.repeat_interleave(n_, m_num, dim=-3)  # [..., M, N, 3]

    # [..., M, N]
    pairwise_dis = torch.sqrt(((m_ - n_) ** 2).sum(dim=-1))

    ret_dis = torch.min(pairwise_dis, dim=-1)[0]  # [..., M]

    return ret_dis


def soft_distance(distance):
    """
    :param distance: [..., M]
    :return: [..., M]
    """
    sigmoid = torch.nn.Sigmoid()
    normalize_factor = 60  # decided by visualization
    return 1 - 2 * (sigmoid(normalize_factor * distance) - 0.5)


def contact_map_of_m_to_n(m, n):
    """
    :param m: [..., M, 3]
    :param n: [..., N, 3]
    :return: [..., M]
    """
    distances = min_distance_from_m_to_n(m, n)  # [..., M]
    distances = soft_distance(distances)
    return distances

def discretize_gt_cm(contact_map, num_bins=10):
    """
    :param contact_map: [B, N], with values within [0, 1]
    :return: [B, N, num_bins]
    """
    bin_boundaries = [i * (1 / num_bins) for i in range(num_bins + 1)]
    bins = []
    contact_map = contact_map.unsqueeze(-1)  # [B, N, 1]
    for i in range(num_bins):
        bins.append(torch.logical_and(contact_map >= bin_boundaries[i],
                                      contact_map < bin_boundaries[i + 1]))
    one_hot = torch.cat(bins, dim=-1).float()  # [B, N, num_bins]
    return one_hot