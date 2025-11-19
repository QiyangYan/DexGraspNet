"""
Last modified date: 2023.02.23
Author: Ruicheng Wang, Jialiang Zhang
Description: Class ObjectModel
"""

import os
import trimesh as tm
import torch
import pytorch3d.structures
import pytorch3d.ops
import numpy as np
import plotly.graph_objects as go
from torchsdf import index_vertices_by_faces, compute_sdf


class ObjectModel:

    def __init__(self, data_root_path, batch_size_each, num_samples=2000, device="cuda"):
        """
        Create a Object Model
        
        Parameters
        ----------
        data_root_path: str
            directory to object meshes
        batch_size_each: int
            batch size for each objects
        num_samples: int
            numbers of object surface points, sampled with fps
        device: str | torch.Device
            device for torch tensors
        """
        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        self.object_code_list = None
        self.object_scale_tensor = None
        self.scene_scale_tensor = None
        self.obj_scale_tensor = None
        self.object_mesh_list = None
        self.object_face_verts_list = None
        self.scale_choice = torch.tensor([1], dtype=torch.float, device=self.device)

    def initialize(self, object_code_list, scene_scale_list=None, obj_scale_list=None,
                   debug_visualize=False, debug_sample_index=0):
        """
        Initialize Object Model with list of objects
        
        Choose scales, load meshes, sample surface points
        
        Parameters
        ----------
        object_code_list: list | str
            list of object codes
        scene_scale_list: list | None
            list of scene scales for each object, if None, will use default scale_choice
        obj_scale_list: list | None
            list of obj scales for each object, if None, will use default scale_choice
        debug_visualize: bool
            若为 True，将在加载完成后可视化指定索引的物体网格并进入调试断点
        debug_sample_index: int
            可视化时使用的 batch 索引（与 get_plotly_data/visualize_mesh 的索引一致）
        """
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        self.object_scale_tensor = []
        self.scene_scale_tensor = []
        self.obj_scale_tensor = []
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        
        # 处理 scene_scale 和 obj_scale
        if scene_scale_list is None:
            scene_scale_list = [None] * len(object_code_list)
        elif not isinstance(scene_scale_list, list):
            scene_scale_list = [scene_scale_list]
        
        if obj_scale_list is None:
            obj_scale_list = [None] * len(object_code_list)
        elif not isinstance(obj_scale_list, list):
            obj_scale_list = [obj_scale_list]
        
        # 确保列表长度匹配
        if len(scene_scale_list) != len(object_code_list):
            scene_scale_list = scene_scale_list[:len(object_code_list)] + [None] * (len(object_code_list) - len(scene_scale_list))
        if len(obj_scale_list) != len(object_code_list):
            obj_scale_list = obj_scale_list[:len(object_code_list)] + [None] * (len(object_code_list) - len(obj_scale_list))
        
        for idx, object_code in enumerate(object_code_list):
            # 处理 scene_scale
            if scene_scale_list[idx] is not None:
                if isinstance(scene_scale_list[idx], (list, np.ndarray)):
                    # 如果是数组，取第一个值（假设所有batch使用相同的scene_scale）
                    scene_scale_val = float(scene_scale_list[idx][0]) if isinstance(scene_scale_list[idx], np.ndarray) else float(scene_scale_list[idx][0])
                else:
                    scene_scale_val = float(scene_scale_list[idx])
                scene_scale_batch = torch.full((self.batch_size_each,), scene_scale_val, dtype=torch.float, device=self.device)
            else:
                scene_scale_batch = self.scale_choice[torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each, ), device=self.device)]
            
            # 处理 obj_scale
            if obj_scale_list[idx] is not None:
                if isinstance(obj_scale_list[idx], np.ndarray):
                    # obj_scale 可能是3D数组 [x, y, z]，取平均值或第一个值
                    if obj_scale_list[idx].ndim == 1:
                        obj_scale_val = float(np.mean(obj_scale_list[idx]))  # 使用平均值
                    else:
                        obj_scale_val = float(obj_scale_list[idx].flat[0])
                elif isinstance(obj_scale_list[idx], (list, tuple)):
                    obj_scale_val = float(np.mean(obj_scale_list[idx]))
                else:
                    obj_scale_val = float(obj_scale_list[idx])
                obj_scale_batch = torch.full((self.batch_size_each,), obj_scale_val, dtype=torch.float, device=self.device)
            else:
                obj_scale_batch = self.scale_choice[torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each, ), device=self.device)]
            
            # 计算总的 object_scale (scene_scale * obj_scale)
            object_scale_batch = scene_scale_batch * obj_scale_batch
            
            self.object_scale_tensor.append(object_scale_batch)
            self.scene_scale_tensor.append(scene_scale_batch)
            self.obj_scale_tensor.append(obj_scale_batch)
            self.object_mesh_list.append(tm.load(os.path.join(self.data_root_path, object_code, "mesh/simplified.obj"), force="mesh", process=False)) # textured_simple.obj
            object_verts = torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device)
            object_faces = torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device)
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
            if self.num_samples != 0:
                vertices = torch.tensor(self.object_mesh_list[-1].vertices, dtype=torch.float, device=self.device)
                faces = torch.tensor(self.object_mesh_list[-1].faces, dtype=torch.float, device=self.device)
                mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
                dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * self.num_samples)
                surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]
                surface_points.to(dtype=float, device=self.device)
                self.surface_points_tensor.append(surface_points)
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)
        self.scene_scale_tensor = torch.stack(self.scene_scale_tensor, dim=0)
        self.obj_scale_tensor = torch.stack(self.obj_scale_tensor, dim=0)
        if self.num_samples != 0:
            self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(self.batch_size_each, dim=0)  # (n_objects * batch_size_each, num_samples, 3)

        if debug_visualize:
            try:
                self.visualize_mesh(
                    i=int(debug_sample_index),
                    title=f"Debug Object Mesh (index={int(debug_sample_index)})",
                    show=True
                )
            except Exception as exc:
                print(f"[ObjectModel] debug_visualize 出现异常: {exc}")
            breakpoint()

    def cal_distance(self, x, with_closest_points=False):
        """
        Calculate signed distances from hand contact points to object meshes and return contact normals
        
        Interiors are positive, exteriors are negative
        
        Use our modified Kaolin package
        
        Parameters
        ----------
        x: (B, `n_contact`, 3) torch.Tensor
            hand contact points
        with_closest_points: bool
            whether to return closest points on object meshes
        
        Returns
        -------
        distance: (B, `n_contact`) torch.Tensor
            signed distances from hand contact points to object meshes, inside is positive
        normals: (B, `n_contact`, 3) torch.Tensor
            contact normal vectors defined by gradient
        closest_points: (B, `n_contact`, 3) torch.Tensor
            contact points on object meshes, returned only when `with_closest_points is True`
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points = []
        scale = self.object_scale_tensor.repeat_interleave(n_points, dim=1)
        x = x / scale.unsqueeze(2)
        for i in range(len(self.object_mesh_list)):
            face_verts = self.object_face_verts_list[i]
            dis, dis_signs, normal, _ = compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance * scale
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = (torch.stack(closest_points) * scale.unsqueeze(2)).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals

    def get_mesh_geometry(self, i, pose=None):
        """
        Get object mesh vertices and faces for visualization.

        Parameters
        ----------
        i: int
            index of data
        pose: (4, 4) matrix or None
            homogeneous transformation matrix to apply to the mesh

        Returns
        -------
        dict
            dictionary containing mesh name, vertices and faces as numpy arrays
        """
        model_index = i // self.batch_size_each
        model_scale = self.object_scale_tensor[model_index, i % self.batch_size_each].detach().cpu().numpy()
        mesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        return {
            'name': f'object_{model_index}',
            'vertices': vertices.astype(np.float32, copy=False),
            'faces': mesh.faces.astype(np.int32, copy=False)
        }
    
    def get_plotly_data(self, i, color='lightgreen', opacity=0.5, pose=None):
        """
        Get visualization data for plotly.graph_objects
        
        Parameters
        ----------
        i: int
            index of data
        color: str
            color of mesh
        opacity: float
            opacity
        pose: (4, 4) matrix
            homogeneous transformation matrix
        
        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        model_index = i // self.batch_size_each
        model_scale = self.object_scale_tensor[model_index, i % self.batch_size_each].detach().cpu().numpy()
        mesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        data = go.Mesh3d(x=vertices[:, 0],y=vertices[:, 1], z=vertices[:, 2], i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2], color=color, opacity=opacity)
        return [data]

    def visualize_mesh(self, i=0, pose=None, color='lightgreen', opacity=0.6,
                       title=None, show=True, save_html_path=None, auto_open=False):
        """
        通过 Plotly 可视化指定 batch 索引对应的物体网格，方便调试 self.object_mesh_list。

        Parameters
        ----------
        i: int
            batch 内的索引（与 get_plotly_data 使用方式一致）
        pose: (4, 4) matrix or None
            可选的齐次变换矩阵，用于在可视化前对网格进行变换
        color: str
            网格颜色
        opacity: float
            网格透明度
        title: str | None
            Plotly figure 的标题，若为 None 会自动生成
        show: bool
            是否立即调用 fig.show() 展示图像
        save_html_path: str | None
            若提供，将把可视化结果保存为 HTML 文件
        auto_open: bool
            保存 HTML 时是否自动在浏览器中打开

        Returns
        -------
        plotly.graph_objects.Figure
            构建好的 Plotly Figure，可供进一步操作
        """
        if not self.object_mesh_list:
            raise RuntimeError("Object meshes 尚未加载，请先调用 initialize().")

        total_samples = len(self.object_mesh_list) * self.batch_size_each
        if i < 0 or i >= total_samples:
            raise IndexError(f"索引 {i} 超出范围 (0 ~ {total_samples - 1})。")

        figure_data = self.get_plotly_data(i=i, color=color, opacity=opacity, pose=pose)
        fig = go.Figure(data=figure_data)

        model_index = i // self.batch_size_each
        batch_index = i % self.batch_size_each
        fig.update_layout(
            scene=dict(aspectmode='data'),
            title=title or f"Object Mesh (model={model_index}, batch={batch_index})",
            showlegend=False
        )

        if save_html_path is not None:
            fig.write_html(save_html_path, include_plotlyjs='cdn', auto_open=auto_open)

        if show:
            fig.show()

        return fig
