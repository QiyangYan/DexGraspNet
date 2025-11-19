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

SHADOWHAND_PART_NAMES = ["palm", "thumb", "index", "middle", "ring", "pinky"]
SHADOWHAND_PART_NAME_TO_ID = {name: idx for idx, name in enumerate(SHADOWHAND_PART_NAMES)}


def _categorize_link_name(link_name: str) -> str:
    if not link_name:
        return "palm"
    lower_name = link_name.lower()
    if any(keyword in lower_name for keyword in ["thumb"]):
        return "thumb"
    if any(keyword in lower_name for keyword in ["index"]):
        return "index"
    if any(keyword in lower_name for keyword in ["middle"]):
        return "middle"
    if any(keyword in lower_name for keyword in ["ring"]):
        return "ring"
    if any(keyword in lower_name for keyword in ["pinky", "little"]):
        return "pinky"
    return "palm"


class HandModel:
    def __init__(self, mjcf_path, mesh_path, contact_points_path, penetration_points_path, n_surface_points=2000, device='cpu'):
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
        self.n_surface_points = n_surface_points
        self.num_hand_parts = len(SHADOWHAND_PART_NAMES)
        
        # Check if this is Shadow Hand by checking if "shadow" is in the path
        is_shadow_hand = "shadow" in mjcf_path.lower() or "shadow" in mesh_path.lower()
        
        # load articulation
        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())
        
        # load contact points and penetration points
        
        contact_points = json.load(open(contact_points_path, 'r')) if contact_points_path is not None else None
        penetration_points = json.load(open(penetration_points_path, 'r')) if penetration_points_path is not None else None
        self.contact_value_current = None
        # build mesh

        self.mesh = {}
        areas = {}
        self.link_part_id_map = {}

        def build_mesh_recurse(body):
            if(len(body.link.visuals) > 0):
                link_name = body.link.name
                part_name = _categorize_link_name(link_name)
                part_id = SHADOWHAND_PART_NAME_TO_ID.get(part_name, SHADOWHAND_PART_NAME_TO_ID["palm"])
                self.link_part_id_map[link_name] = part_id
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    # import ipdb;ipdb.set_trace()
                    scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
                    # Skip unsupported or collision-only geometry types
                    if visual.geom_type in ["box"]:
                        # Only load box.obj for Shadow Hand
                        if is_shadow_hand:
                            # link_mesh = tm.primitives.Box(extents=2 * visual.geom_param)
                            link_mesh = tm.load_mesh(os.path.join(mesh_path, 'box.obj'), process=False)
                            link_mesh.vertices *= visual.geom_param.cpu().numpy()
                            # print(visual.geom_param)
                            # import ipdb; ipdb.set_trace()
                        else:
                            # For non-Shadow Hand, skip box/sphere geometries
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
                # Skip if no valid mesh was loaded
                if len(link_vertices) > 0:
                    link_vertices = torch.cat(link_vertices, dim=0)
                    link_faces = torch.cat(link_faces, dim=0)
                    contact_candidates = torch.tensor(contact_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3) if contact_points is not None else None
                    penetration_keypoints = torch.tensor(penetration_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3) if penetration_points is not None else None
                    self.mesh[link_name] = {
                        'vertices': link_vertices,
                        'faces': link_faces,
                        'contact_candidates': contact_candidates,
                        'penetration_keypoints': penetration_keypoints,
                        'part_id': part_id,
                    }
                    link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
                    self.mesh[link_name]['face_verts'] = link_face_verts
                    # self.mesh[link_name]['geom_param'] = body.link.visuals[0].geom_param
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

        self.penetration_keypoints = [self.mesh[link_name]['penetration_keypoints'] for link_name in self.mesh]
        self.global_index_to_link_index_penetration = sum([[i] * len(penetration_keypoints) for i, penetration_keypoints in enumerate(self.penetration_keypoints)], [])
        self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        self.global_index_to_link_index_penetration = torch.tensor(self.global_index_to_link_index_penetration, dtype=torch.long, device=device)
        self.n_keypoints = self.penetration_keypoints.shape[0]

        # parameters

        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None
        
        # For contact map computation (CMapAdam style)
        self.object_point_cloud = None
        self.object_normal_cloud = None
        self.contact_value_goal = None
        self.contact_value_goal_parts = None
        self.contact_value_current_parts = None
        self.contact_loss_per_part = None
        self.debug_cmap = False
        self.contact_value_part_names = list(SHADOWHAND_PART_NAMES)
        self.contact_map_goal_per_sample = []
        self.contact_value_current_per_sample = []
        self.contact_value_current_parts_per_sample = []
        self.contact_map_current_part_names = []
        
        # For tensorboard logging
        self.logger = None
        self.current_step = None

    def allocate_contact_map_goals(self, total_batch_size: int):
        """
        Allocate storage for per-sample contact map goals.
        """
        total_batch_size = int(max(total_batch_size, 0))
        self.contact_map_goal_per_sample = [None] * total_batch_size
        self.contact_value_current_per_sample = [None] * total_batch_size
        self.contact_value_current_parts_per_sample = [None] * total_batch_size
        self.contact_map_current_part_names = [list(self.contact_value_part_names) for _ in range(total_batch_size)]

    def _ensure_cmap_goal_capacity(self, capacity: int):
        if capacity <= 0:
            return
        current = len(self.contact_map_goal_per_sample)
        if current < capacity:
            pad = capacity - current
            self.contact_map_goal_per_sample.extend([None] * pad)
            self.contact_value_current_per_sample.extend([None] * pad)
            self.contact_value_current_parts_per_sample.extend([None] * pad)
            self.contact_map_current_part_names.extend(
                [list(self.contact_value_part_names) for _ in range(pad)]
            )

    def _to_device_tensor(self, data, dtype=torch.float32):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            tensor = data.to(self.device, dtype=dtype)
        else:
            tensor = torch.tensor(data, dtype=dtype, device=self.device)
        return tensor.detach()

    def _clone_goal_entry(self, entry):
        if entry is None:
            return None
        return {
            'points': entry['points'].clone().detach(),
            'normals': entry['normals'].clone().detach() if entry['normals'] is not None else None,
            'values': entry['values'].clone().detach(),
            'parts': entry['parts'].clone().detach() if entry['parts'] is not None else None,
            'part_names': list(entry.get('part_names', SHADOWHAND_PART_NAMES)),
        }

    def _parse_contact_map_goal_input(self, contact_map_goal, contact_map_goal_parts=None, part_names=None):
        if contact_map_goal is None:
            return None

        local_part_names = part_names
        obj_points = None
        obj_normals = None
        contact_values = None
        parts_tensor = None

        if isinstance(contact_map_goal, dict):
            obj_points = contact_map_goal.get('object_point_cloud', None)
            obj_normals = contact_map_goal.get('object_normal_cloud', None)
            contact_values = contact_map_goal.get('contact_value', None)
            if contact_map_goal_parts is None:
                contact_map_goal_parts = contact_map_goal.get('contact_value_parts', None)
            if local_part_names is None:
                local_part_names = contact_map_goal.get('contact_value_part_names', None)
        else:
            goal_tensor = self._to_device_tensor(contact_map_goal)
            if goal_tensor.ndim != 2 or goal_tensor.shape[1] < 6:
                raise ValueError("contact_map_goal 应为形状 (N, 7) 或包含 point/normal/value 的字典。")
            obj_points = goal_tensor[:, :3]
            obj_normals = goal_tensor[:, 3:6]
            if goal_tensor.shape[1] < 7:
                raise ValueError("contact_map_goal 缺少 contact value 列。")
            contact_values = goal_tensor[:, 6]

        obj_points = self._to_device_tensor(obj_points)
        if obj_points is None or obj_points.numel() == 0:
            return None
        obj_normals = self._to_device_tensor(obj_normals)
        contact_values = self._to_device_tensor(contact_values)
        if contact_values is None:
            raise ValueError("contact_map_goal 缺少 contact value 数据。")
        parts_tensor = self._to_device_tensor(contact_map_goal_parts)

        if parts_tensor is not None and parts_tensor.dim() == 1:
            parts_tensor = parts_tensor.unsqueeze(0)

        target_len = obj_points.shape[0]
        min_len = target_len
        min_len = min(min_len, contact_values.shape[0])
        if obj_normals is not None:
            min_len = min(min_len, obj_normals.shape[0])
        if parts_tensor is not None:
            if parts_tensor.dim() != 2:
                raise ValueError(f"contact_map_goal_parts 形状应为 (num_parts, N)，当前为 {parts_tensor.shape}")
            min_len = min(min_len, parts_tensor.shape[1])

        if min_len <= 0:
            return None

        obj_points = obj_points[:min_len]
        contact_values = contact_values[:min_len]
        if obj_normals is not None:
            obj_normals = obj_normals[:min_len]
        if parts_tensor is not None:
            parts_tensor = parts_tensor[:, :min_len]

        if local_part_names is not None:
            part_name_list = list(local_part_names)
        else:
            part_name_list = list(self.contact_value_part_names)
        if not part_name_list:
            part_name_list = list(SHADOWHAND_PART_NAMES)

        return {
            'points': obj_points,
            'normals': obj_normals,
            'values': contact_values,
            'parts': parts_tensor,
            'part_names': part_name_list,
        }

    def set_contact_map_goal(self,
                             contact_map_goal,
                             contact_map_goal_parts=None,
                             sample_idx=None,
                             part_names=None):
        """
        为指定样本（或全部样本）设置 contact map goal。

        Parameters
        ----------
        contact_map_goal: dict | np.ndarray | torch.Tensor
            支持字典形式（含 object_point_cloud 等键）或 (N,7) 形式的张量/数组
        contact_map_goal_parts: optional
            若 contact_map_goal 未包含 contact_map_object_parts，可通过此参数额外提供
        sample_idx: int | Sequence[int] | None
            目标样本索引。当为 None 时，默认将 goal 应用于当前批中的所有样本
        part_names: list | tuple | None
            自定义手部部位名称
        """
        goal_entry = self._parse_contact_map_goal_input(contact_map_goal,
                                                        contact_map_goal_parts,
                                                        part_names)

        if isinstance(sample_idx, (list, tuple)):
            target_indices = [int(idx) for idx in sample_idx]
        elif sample_idx is None:
            total = self.hand_pose.shape[0] if self.hand_pose is not None else 1
            target_indices = list(range(total))
        else:
            target_indices = [int(sample_idx)]

        if not target_indices:
            return

        max_index = max(target_indices) + 1
        self._ensure_cmap_goal_capacity(max_index)

        for idx in target_indices:
            cloned_entry = self._clone_goal_entry(goal_entry)
            self.contact_map_goal_per_sample[idx] = cloned_entry
            if cloned_entry is not None and cloned_entry.get('part_names'):
                self.contact_map_current_part_names[idx] = list(cloned_entry['part_names'])

        reference_entry = self.contact_map_goal_per_sample[target_indices[0]]
        if reference_entry is not None:
            self.object_point_cloud = reference_entry['points']
            self.object_normal_cloud = reference_entry['normals']
            self.contact_value_goal = reference_entry['values']
            self.contact_value_goal_parts = reference_entry['parts']
            if reference_entry['part_names']:
                self.contact_value_part_names = list(reference_entry['part_names'])
        else:
            if target_indices == list(range(len(self.contact_map_goal_per_sample))):
                self.object_point_cloud = None
                self.object_normal_cloud = None
                self.contact_value_goal = None
                self.contact_value_goal_parts = None

    def get_contact_map_goal(self, sample_idx: int):
        if sample_idx < 0 or sample_idx >= len(self.contact_map_goal_per_sample):
            return None
        return self.contact_map_goal_per_sample[sample_idx]

    def get_contact_map_current(self, sample_idx: int):
        if sample_idx < 0 or sample_idx >= len(self.contact_value_current_per_sample):
            return None
        return self.contact_value_current_per_sample[sample_idx]

    def get_contact_map_current_parts(self, sample_idx: int):
        if sample_idx < 0 or sample_idx >= len(self.contact_value_current_parts_per_sample):
            return None
        return self.contact_value_current_parts_per_sample[sample_idx]

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

        # surface points in hand frame
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            if link_name in ["hand_base"]:
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
                # compute the distance of local point and axis point 
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
        valid_links = [ln for ln in self.mesh if ln not in ['hand_base']]
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
        # import ipdb;ipdb.set_trace()
        return E_spen.sum((1,2))

    def get_surface_points(self):
        """
        Get surface points
        
        Returns
        -------
        points: (N, `n_surface_points`, 3)
            surface points
        """
        points, _ = self._get_surface_points_with_part_ids()
        return points

    def _get_surface_points_with_part_ids(self):
        """
        Get surface points along with hand-part ids.
        Mimics the sampling strategy used in store_dexonomy_retarget.get_hand_surface_points_from_handmodel:
            - Palm:fingers = 5:3 (per finger)
            - Points for each part are distributed evenly across its links
        Returns
        -------
        points: (B, `n_surface_points`, 3)
        part_ids: (B, `n_surface_points`)
        """
        batch_size = self.global_translation.shape[0]

        # Build link groups by part
        part_to_links = {idx: [] for idx in range(len(SHADOWHAND_PART_NAMES))}
        for link_name, meta in self.mesh.items():
            n_surface_points = meta['surface_points'].shape[0]
            if n_surface_points == 0:
                continue
            part_id = meta.get('part_id', SHADOWHAND_PART_NAME_TO_ID["palm"])
            part_to_links.setdefault(part_id, [])
            part_to_links[part_id].append(link_name)

        total_target = getattr(self, "n_surface_points", None)
        if total_target is None or total_target <= 0:
            total_target = sum(self.mesh[ln]['surface_points'].shape[0] for ln in self.mesh)

        # Compute target samples per part (palm: 5k, each finger: 3k)
        palm_total_samples = int(total_target / 4) if total_target > 0 else 0
        finger_total_samples = int(3 * total_target / 20) if total_target > 0 else 0

        part_per_link_samples = {}
        for part_id, part_name in enumerate(SHADOWHAND_PART_NAMES):
            links = part_to_links.get(part_id, [])
            link_count = len(links)
            if link_count == 0:
                part_per_link_samples[part_id] = 0
                continue
            if part_name == "palm":
                if palm_total_samples <= 0:
                    part_per_link_samples[part_id] = 0
                else:
                    part_per_link_samples[part_id] = max(1, palm_total_samples // link_count)
            else:
                if finger_total_samples <= 0:
                    part_per_link_samples[part_id] = 0
                else:
                    part_per_link_samples[part_id] = max(1, finger_total_samples // link_count)

        sampled_points = []
        sampled_part_ids = []
        device = self.device

        for part_id, links in part_to_links.items():
            if len(links) == 0:
                continue
            samples_per_link = part_per_link_samples.get(part_id, 0)
            if samples_per_link <= 0:
                continue

            for link_name in links:
                surface_pts_local = self.mesh[link_name]['surface_points']
                n_available = surface_pts_local.shape[0]
                if n_available == 0:
                    continue

                if n_available >= samples_per_link:
                    idx = torch.randperm(n_available, device=device)[:samples_per_link]
                else:
                    idx = torch.randint(0, n_available, (samples_per_link,), device=device)

                selected_local = surface_pts_local[idx]
                transformed = self.current_status[link_name].transform_points(selected_local)
                if transformed.dim() == 2:
                    transformed = transformed.unsqueeze(0)
                if transformed.shape[0] != batch_size:
                    transformed = transformed.expand(batch_size, samples_per_link, 3)
                transformed = transformed.to(device)
                transformed = transformed @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)

                sampled_points.append(transformed)
                sampled_part_ids.append(torch.full((batch_size, samples_per_link), part_id, dtype=torch.long, device=device))

        if len(sampled_points) == 0:
            empty_points = torch.zeros(batch_size, 0, 3, dtype=torch.float, device=device)
            empty_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            return empty_points, empty_ids

        points = torch.cat(sampled_points, dim=1)
        part_ids = torch.cat(sampled_part_ids, dim=1)
        current_num = points.shape[1]

        if total_target > 0 and current_num != total_target:
            if current_num > total_target:
                idx = torch.randperm(current_num, device=device)[:total_target]
            else:
                idx = torch.randint(0, current_num, (total_target,), device=device)
            points = points[:, idx]
            part_ids = part_ids[:, idx]

        return points.contiguous(), part_ids.contiguous()

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

    def set_cmap_debug(self, flag: bool = True):
        """
        Enable or disable debugging utilities for contact map optimization.
        When enabled, _compute_cmap_energy_align_dist will pause execution and
        show Open3D visualizations of current vs goal contact maps.
        """
        self.debug_cmap = flag

    @staticmethod
    def _map_values_to_rgb(values: np.ndarray) -> np.ndarray:
        """
        Convert normalized contact values (any real array) to RGB colors using the
        same blue→cyan→green→yellow→red scheme.
        """
        if values.ndim != 1:
            values = values.reshape(-1)
        finite_mask = np.isfinite(values)
        colors = np.zeros((values.shape[0], 3), dtype=np.float32)
        if not np.any(finite_mask):
            return colors + np.array([0.0, 0.0, 1.0], dtype=np.float32)

        v = values[finite_mask]
        min_v = np.min(v)
        max_v = np.max(v)
        denom = max(max_v - min_v, 1e-8)
        norm = np.clip((values - min_v) / denom, 0.0, 1.0)

        for idx, val in enumerate(norm):
            if val < 0.25:
                r, g, b = 0.0, val * 4.0, 1.0
            elif val < 0.5:
                r, g, b = 0.0, 1.0, 1.0 - (val - 0.25) * 4.0
            elif val < 0.75:
                r, g, b = (val - 0.5) * 4.0, 1.0, 0.0
            else:
                r, g, b = 1.0, 1.0 - (val - 0.75) * 4.0, 0.0
            colors[idx] = [r, g, b]
        return colors

    def _visualize_contact_map_o3d(self, object_points: np.ndarray,
                                   values: np.ndarray,
                                   window_name: str):
        """
        Visualize a single contact map with Open3D.
        """
        try:
            import open3d as o3d
        except ImportError:
            print("[CMap Debug] open3d 未安装，无法可视化。请先安装 open3d 包。")
            return

        if object_points.ndim != 2 or object_points.shape[1] != 3:
            raise ValueError(f"object_points 形状需要为 (N, 3)，当前为 {object_points.shape}")
        if values.shape[0] != object_points.shape[0]:
            raise ValueError("object_points 与 contact map 长度不匹配")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points.astype(np.float32))
        colors = self._map_values_to_rgb(values.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name=window_name)
    
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
    
    def get_cmap_loss(self, energy_func_name='align_dist'):
        """
        Calculate contact map loss using CMapAdam-style computation
        
        Parameters
        ----------
        energy_func_name: str
            'align_dist' or 'euclidean_dist'
            
        Returns
        -------
        E_cmap: (B,) torch.Tensor
            contact map energy for each sample in batch
        """
        if self.hand_pose is None:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        batch_size = self.hand_pose.shape[0]
        self._ensure_cmap_goal_capacity(batch_size)

        if len(self.contact_map_goal_per_sample) == 0:
            return torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        hand_surface_points_all, hand_surface_part_ids_all = self._get_surface_points_with_part_ids()

        losses = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        per_part_losses = []
        per_part_mse = []
        valid_indices = []

        current_values_list = [None] * batch_size
        current_parts_list = [None] * batch_size

        for idx in range(batch_size):
            goal_entry = self.get_contact_map_goal(idx)
            if goal_entry is None:
                continue

            obj_points = goal_entry.get('points', None)
            goal_values = goal_entry.get('values', None)
            if obj_points is None or goal_values is None or obj_points.numel() == 0 or goal_values.numel() == 0:
                continue

            obj_normals = goal_entry.get('normals', None)
            goal_parts = goal_entry.get('parts', None)

            local_energy_func = energy_func_name
            if local_energy_func == 'align_dist' and (obj_normals is None or obj_normals.numel() == 0):
                local_energy_func = 'euclidean_dist'

            contact_current, contact_current_parts = self.compute_contact_map_for_sample(
                sample_idx=idx,
                object_points=obj_points,
                object_normals=obj_normals,
                energy_func_name=local_energy_func,
                hand_surface_points=hand_surface_points_all,
                hand_surface_part_ids=hand_surface_part_ids_all
            )

            pred_len = contact_current.shape[0]
            goal_len = goal_values.shape[0]
            effective_len = min(pred_len, goal_len)
            if goal_parts is not None:
                effective_len = min(effective_len, goal_parts.shape[1])
            if obj_normals is not None:
                effective_len = min(effective_len, obj_normals.shape[0])
            effective_len = min(effective_len, obj_points.shape[0])

            if effective_len <= 0:
                continue

            if pred_len != effective_len:
                contact_current = contact_current[:effective_len]
                contact_current_parts = contact_current_parts[:, :effective_len]
            if goal_len != effective_len:
                goal_values = goal_values[:effective_len]
                goal_entry['values'] = goal_values
            if obj_points.shape[0] != effective_len:
                goal_entry['points'] = goal_entry['points'][:effective_len]
            if obj_normals is not None and obj_normals.shape[0] != effective_len:
                goal_entry['normals'] = goal_entry['normals'][:effective_len]
                obj_normals = goal_entry['normals']

            if goal_parts is not None:
                if goal_parts.shape[1] != effective_len:
                    goal_parts = goal_parts[:, :effective_len]
                    goal_entry['parts'] = goal_parts
            else:
                goal_parts = goal_values.view(1, -1).expand(self.num_hand_parts, -1)
                goal_entry['parts'] = goal_parts

            part_loss = torch.abs(contact_current_parts - goal_parts).mean(dim=1)
            part_mse = torch.mean((contact_current_parts - goal_parts) ** 2, dim=1)

            losses[idx] = part_loss.mean()
            per_part_losses.append(part_loss)
            per_part_mse.append(part_mse)
            valid_indices.append(idx)

            current_values_list[idx] = contact_current.detach()
            current_parts_list[idx] = contact_current_parts.detach()

        if per_part_losses:
            stacked_losses = torch.stack(per_part_losses, dim=0)
            stacked_mse = torch.stack(per_part_mse, dim=0)
            self.contact_loss_per_part = stacked_losses.detach()
            self.contact_loss_per_part_mean = stacked_losses.mean(dim=0).detach()
            self.contact_loss_per_part_mse = stacked_mse.detach()
            self.contact_loss_per_part_mse_mean = stacked_mse.mean(dim=0).detach()

            first_length = None
            consistent_length = True
            for idx in valid_indices:
                parts_tensor = current_parts_list[idx]
                if parts_tensor is None:
                    consistent_length = False
                    break
                length = parts_tensor.shape[1]
                if first_length is None:
                    first_length = length
                elif length != first_length:
                    consistent_length = False
                    break

            if consistent_length and first_length is not None:
                current_parts_stack = torch.stack(
                    [current_parts_list[idx] for idx in valid_indices],
                    dim=0
                )
                goal_parts_stack = torch.stack(
                    [self.contact_map_goal_per_sample[idx]['parts'] for idx in valid_indices],
                    dim=0
                )
                self._print_cmap_summary(stacked_losses, current_parts_stack, goal_parts_stack)
        else:
            self.contact_loss_per_part = None
            self.contact_loss_per_part_mean = None
            self.contact_loss_per_part_mse = None
            self.contact_loss_per_part_mse_mean = None

        self.contact_value_current_per_sample = current_values_list
        self.contact_value_current_parts_per_sample = current_parts_list
        self.contact_value_current = current_values_list
        self.contact_value_current_parts = current_parts_list

        return losses

    def _print_cmap_summary(self, energy_contact_parts, contact_value_current_parts, goal_parts):
        """
        Print and log contact map summary to tensorboard
        
        Parameters
        ----------
        energy_contact_parts: (B, num_parts) torch.Tensor
            contact map loss for each part
        contact_value_current_parts: (B, num_parts, N_obj) torch.Tensor
            predicted contact values for each part
        goal_parts: (B, num_parts, N_obj) torch.Tensor
            target contact values for each part
        """
        with torch.no_grad():
            loss_mean = energy_contact_parts.mean(dim=0).detach().cpu()
            pred_mean = contact_value_current_parts.mean(dim=2).mean(dim=0).detach().cpu()
            goal_mean = goal_parts.mean(dim=2).mean(dim=0).detach().cpu()
        
        # Print to console
        # for idx, name in enumerate(SHADOWHAND_PART_NAMES):
        #     print(f"[CMap] {name}: 平均Loss={loss_mean[idx]:.4f}, 预测均值={pred_mean[idx]:.4f}, 目标均值={goal_mean[idx]:.4f}")
        
        # Log to tensorboard if logger is available
        if self.logger is not None and self.current_step is not None:
            for idx, name in enumerate(SHADOWHAND_PART_NAMES):
                # Log loss for each part (E curve)
                self.logger.writer.add_scalar(f'CMap/Loss/{name}', loss_mean[idx].item(), self.current_step)
                # Log predicted mean contact value for each part
                self.logger.writer.add_scalar(f'CMap/PredMean/{name}', pred_mean[idx].item(), self.current_step)
                # Log goal mean contact value for each part
                self.logger.writer.add_scalar(f'CMap/GoalMean/{name}', goal_mean[idx].item(), self.current_step)
    
    def visualize_cmap(self, object_pc, contact_value, i=0, show_hand=True, 
                       opacity_hand=0.5, color_hand='lightblue', save_path=None):
        """
        Visualize contact map on object point cloud using plotly
        
        Color scheme: Blue (no contact) -> Red (high contact)
        
        Parameters
        ----------
        object_pc: (N, 3) torch.Tensor or numpy.ndarray
            object point cloud
        contact_value: (N,) torch.Tensor or numpy.ndarray
            contact value for each point (0-1), where 0 is no contact (blue) 
            and 1 is high contact (red)
        i: int
            batch index to visualize
        show_hand: bool
            whether to show hand mesh
        opacity_hand: float
            opacity of hand mesh
        color_hand: str
            color of hand mesh
        save_path: str or None
            if provided, save the visualization to this path (e.g., 'output.html')
        
        Returns
        -------
        fig: plotly.graph_objects.Figure
            plotly figure object
        """
        # Convert to numpy if needed
        if isinstance(object_pc, torch.Tensor):
            object_pc = object_pc.detach().cpu().numpy()
        if isinstance(contact_value, torch.Tensor):
            contact_value = contact_value.detach().cpu().numpy()
        
        # Ensure contact_value is in range [0, 1]
        contact_value = np.clip(contact_value, 0, 1)
        
        # Create contact color map: blue (no contact) -> red (high contact)
        # Blue: RGB(0, 0, 255), Red: RGB(255, 0, 0)
        contact_colors = [
            f"rgb({int(255 * val)},{0},{int(255 * (1 - val))})" 
            for val in contact_value
        ]
        
        # Create point cloud with contact map
        point_cloud_trace = go.Scatter3d(
            x=object_pc[:, 0],
            y=object_pc[:, 1],
            z=object_pc[:, 2],
            mode='markers',
            marker=dict(
                color=contact_colors,
                size=3.5,
                opacity=1
            ),
            name='Object with Contact Map'
        )
        
        data = [point_cloud_trace]
        
        # Add hand mesh if requested
        if show_hand:
            hand_data = self.get_plotly_data(i=i, opacity=opacity_hand, 
                                            color=color_hand, with_contact_points=False)
            data.extend(hand_data)
        
        # Create figure
        fig = go.Figure(data=data)
        
        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'
            ),
            title='Contact Map Visualization',
            showlegend=True
        )
        
        # Save if path provided
        if save_path is not None:
            fig.write_html(save_path)
            print(f"Visualization saved to {save_path}")
        
        return fig


    def _compute_cmap_energy_align_dist(self):
        """
        Compute contact map energy using align distance (CMapAdam style)
        """
        hand_surface_points, hand_surface_part_ids = self._get_surface_points_with_part_ids()
        batch_size, npts_hand, _ = hand_surface_points.shape
        npts_object = self.object_point_cloud.size(0)

        batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).expand(batch_size, -1, -1)
        batch_object_normal_cloud = self.object_normal_cloud.unsqueeze(0).expand(batch_size, -1, -1)

        hand_surface_points_exp = hand_surface_points.unsqueeze(1).expand(-1, npts_object, -1, -1)
        object_points_exp = batch_object_point_cloud.unsqueeze(2).expand(-1, npts_object, npts_hand, -1)
        object_normals_exp = batch_object_normal_cloud.unsqueeze(2).expand(-1, npts_object, npts_hand, -1)

        diff = hand_surface_points_exp - object_points_exp
        object_hand_dist = diff.norm(dim=3)
        object_hand_align = (diff * object_normals_exp).sum(dim=3) / (object_hand_dist + 1e-5)

        object_hand_align_dist = object_hand_dist * torch.exp(2 * (1 - object_hand_align))
        contact_dist = torch.sqrt(object_hand_align_dist.min(dim=2)[0])
        contact_value_current = 1 - 2 * (torch.sigmoid(10 * contact_dist) - 0.5)

        if self.debug_cmap:
            try:
                current_np = contact_value_current[0].detach().cpu().numpy()
                goal_np = self.contact_value_goal.detach().cpu().numpy() if self.contact_value_goal is not None else None
                object_np = self.object_point_cloud.detach().cpu().numpy()
                self._visualize_contact_map_o3d(object_np, current_np, "CMap Debug - Current")
                if goal_np is not None:
                    self._visualize_contact_map_o3d(object_np, goal_np, "CMap Debug - Goal")
            except Exception as exc:
                print(f"[CMap Debug] 可视化失败: {exc}")
            try:
                import ipdb; ipdb.set_trace()
            except ImportError:
                import pdb; pdb.set_trace()

        hand_surface_part_ids_exp = hand_surface_part_ids.unsqueeze(1).expand(-1, npts_object, -1)
        contact_value_current_parts = []
        for part_id in range(self.num_hand_parts):
            part_mask = (hand_surface_part_ids_exp == part_id)
            align_dist_part = object_hand_align_dist.masked_fill(~part_mask, float('inf'))
            contact_dist_part = torch.sqrt(align_dist_part.min(dim=2)[0])
            contact_value_part = 1 - 2 * (torch.sigmoid(10 * contact_dist_part) - 0.5)
            contact_value_current_parts.append(contact_value_part.unsqueeze(1))

        contact_value_current_parts = torch.cat(contact_value_current_parts, dim=1)

        if self.contact_value_goal_parts is not None:
            goal_parts = self.contact_value_goal_parts.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            goal_parts = self.contact_value_goal.view(1, 1, -1).expand(batch_size, self.num_hand_parts, -1)

        energy_contact_parts = torch.abs(contact_value_current_parts - goal_parts).mean(dim=2)
        energy_contact = energy_contact_parts.mean(dim=1)

        return energy_contact, energy_contact_parts, contact_value_current, contact_value_current_parts, goal_parts
    
    def _compute_cmap_energy_euclidean_dist(self):
        """
        Compute contact map energy using euclidean distance (CMapAdam style)
        """
        hand_surface_points, hand_surface_part_ids = self._get_surface_points_with_part_ids()
        batch_size, npts_hand, _ = hand_surface_points.shape
        npts_object = self.object_point_cloud.size(0)

        batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).expand(batch_size, -1, -1)

        hand_surface_points_exp = hand_surface_points.unsqueeze(1).expand(-1, npts_object, -1, -1)
        object_points_exp = batch_object_point_cloud.unsqueeze(2).expand(-1, npts_object, npts_hand, -1)

        object_hand_dist = (hand_surface_points_exp - object_points_exp).norm(dim=3)
        contact_dist = object_hand_dist.min(dim=2)[0]
        contact_value_current = 1 - 2 * (torch.sigmoid(100 * contact_dist) - 0.5)

        hand_surface_part_ids_exp = hand_surface_part_ids.unsqueeze(1).expand(-1, npts_object, -1)
        contact_value_current_parts = []
        for part_id in range(self.num_hand_parts):
            part_mask = (hand_surface_part_ids_exp == part_id)
            dist_part = object_hand_dist.masked_fill(~part_mask, float('inf'))
            contact_dist_part = dist_part.min(dim=2)[0]
            contact_value_part = 1 - 2 * (torch.sigmoid(100 * contact_dist_part) - 0.5)
            contact_value_current_parts.append(contact_value_part.unsqueeze(1))

        contact_value_current_parts = torch.cat(contact_value_current_parts, dim=1)

        if self.contact_value_goal_parts is not None:
            goal_parts = self.contact_value_goal_parts.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            goal_parts = self.contact_value_goal.view(1, 1, -1).expand(batch_size, self.num_hand_parts, -1)

        energy_contact_parts = torch.abs(contact_value_current_parts - goal_parts).mean(dim=2)
        energy_contact = energy_contact_parts.mean(dim=1)

        return energy_contact, energy_contact_parts, contact_value_current, contact_value_current_parts, goal_parts

    def compute_contact_map_for_sample(self, sample_idx, object_points, object_normals=None,
                                       energy_func_name='align_dist',
                                       hand_surface_points=None,
                                       hand_surface_part_ids=None):
        """
        计算指定 batch 样本在给定物体点云上的 contact map（不修改 HandModel 内部状态）。

        Parameters
        ----------
        sample_idx: int
            batch 内的样本索引
        object_points: array-like | torch.Tensor
            物体点云，形状为 (N, 3)
        object_normals: array-like | torch.Tensor | None
            物体法向量，align_dist 模式下必需，形状为 (N, 3)
        energy_func_name: str
            'align_dist' 或 'euclidean_dist'
        hand_surface_points: torch.Tensor | None
            可复用的手部表面点集合 (B, M, 3)，若为 None 将自动重新计算
        hand_surface_part_ids: torch.Tensor | None
            与 hand_surface_points 对应的部位索引 (B, M)

        Returns
        -------
        contact_value: (N,) torch.Tensor
            预测的整体 contact map
        contact_value_parts: (num_parts, N) torch.Tensor
            各手部部位的 contact map
        """
        if self.global_translation is None or self.global_rotation is None:
            raise RuntimeError("HandModel 尚未设置手部参数，请先调用 set_parameters().")

        batch_size = self.global_translation.shape[0]
        if sample_idx < 0 or sample_idx >= batch_size:
            raise IndexError(f"sample_idx={sample_idx} 超出范围 (0 ~ {batch_size - 1})。")

        if isinstance(object_points, torch.Tensor):
            obj_points = object_points.to(self.device, dtype=torch.float32)
        else:
            obj_points = torch.tensor(object_points, dtype=torch.float32, device=self.device)

        if obj_points.ndim != 2 or obj_points.shape[1] != 3:
            raise ValueError(f"object_points 形状应为 (N, 3)，当前为 {obj_points.shape}")

        num_obj_pts = obj_points.shape[0]
        if num_obj_pts == 0:
            empty_current = torch.zeros(0, dtype=torch.float32, device=self.device)
            empty_parts = torch.zeros(self.num_hand_parts, 0, dtype=torch.float32, device=self.device)
            return empty_current, empty_parts

        if hand_surface_points is None or hand_surface_part_ids is None:
            hand_surface_points, hand_surface_part_ids = self._get_surface_points_with_part_ids()
        else:
            if isinstance(hand_surface_points, torch.Tensor):
                hand_surface_points = hand_surface_points.to(self.device, dtype=torch.float32)
            else:
                hand_surface_points = torch.tensor(hand_surface_points, dtype=torch.float32, device=self.device)
            if isinstance(hand_surface_part_ids, torch.Tensor):
                hand_surface_part_ids = hand_surface_part_ids.to(self.device, dtype=torch.long)
            else:
                hand_surface_part_ids = torch.tensor(hand_surface_part_ids, dtype=torch.long, device=self.device)

        if sample_idx >= hand_surface_points.shape[0]:
            raise IndexError(f"hand_surface_points 不包含 sample_idx={sample_idx} 的数据。")

        hand_surface_points = hand_surface_points[sample_idx:sample_idx + 1]  # (1, M, 3)
        hand_surface_part_ids = hand_surface_part_ids[sample_idx:sample_idx + 1]  # (1, M)

        hand_surface_points_exp = hand_surface_points.unsqueeze(1).expand(-1, num_obj_pts, -1, -1)
        object_points_exp = obj_points.unsqueeze(0).unsqueeze(2).expand_as(hand_surface_points_exp)

        if energy_func_name == 'align_dist':
            if object_normals is None:
                raise ValueError("energy_func_name='align_dist' 时必须提供 object_normals。")
            if isinstance(object_normals, torch.Tensor):
                obj_normals = object_normals.to(self.device, dtype=torch.float32)
            else:
                obj_normals = torch.tensor(object_normals, dtype=torch.float32, device=self.device)
            if obj_normals.shape != obj_points.shape:
                raise ValueError(f"object_normals 形状必须与 object_points 一致，当前为 {obj_normals.shape} vs {obj_points.shape}.")
            object_normals_exp = obj_normals.unsqueeze(0).unsqueeze(2).expand_as(hand_surface_points_exp)
            diff = hand_surface_points_exp - object_points_exp
            object_hand_dist = diff.norm(dim=3)
            object_hand_align = (diff * object_normals_exp).sum(dim=3) / (object_hand_dist + 1e-5)
            metric = object_hand_dist * torch.exp(2 * (1 - object_hand_align))
            sigmoid_scale = 10.0
            contact_dist = torch.sqrt(metric.min(dim=2)[0])
        elif energy_func_name == 'euclidean_dist':
            diff = hand_surface_points_exp - object_points_exp
            metric = diff.norm(dim=3)
            sigmoid_scale = 100.0
            contact_dist = metric.min(dim=2)[0]
        else:
            raise ValueError(f"Unknown energy_func_name: {energy_func_name}")

        contact_value = 1 - 2 * (torch.sigmoid(sigmoid_scale * contact_dist) - 0.5)

        hand_surface_part_ids_exp = hand_surface_part_ids.unsqueeze(1).expand(-1, num_obj_pts, -1)
        contact_value_parts = []
        for part_id in range(self.num_hand_parts):
            part_mask = hand_surface_part_ids_exp == part_id
            if part_mask.any():
                part_metric = metric.clone()
                part_metric = part_metric.masked_fill(~part_mask, float('inf'))
                if energy_func_name == 'align_dist':
                    part_dist = torch.sqrt(part_metric.min(dim=2)[0])
                else:
                    part_dist = part_metric.min(dim=2)[0]
                part_value = 1 - 2 * (torch.sigmoid(sigmoid_scale * part_dist) - 0.5)
            else:
                part_value = torch.zeros(1, num_obj_pts, dtype=torch.float32, device=self.device)
            contact_value_parts.append(part_value.unsqueeze(1))

        contact_value_parts = torch.cat(contact_value_parts, dim=1)  # (1, num_parts, N)
        return contact_value.squeeze(0), contact_value_parts.squeeze(0)

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