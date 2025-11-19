"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: Class Annealing optimizer
"""

import torch

def clip_around_init(x: torch.Tensor, x0: torch.Tensor, rng) -> torch.Tensor:
    """
    Clamp x to [x0 - rng, x0 + rng] element-wise.

    x, x0: (B, D) tensors
    rng: scalar, list/tuple/np/tensor of length D, or (1, D) tensor
    """
    x  = torch.as_tensor(x)
    x0 = torch.as_tensor(x0, device=x.device, dtype=x.dtype)
    r  = torch.as_tensor(rng, device=x.device, dtype=x.dtype)

    if r.ndim == 0:                      # scalar -> broadcast
        r = r.expand_as(x)
    elif r.ndim == 1:                    # (D,) -> (B,D)
        r = r.view(1, -1).expand_as(x)
    else:                                # (1,D) or (B,D) -> (B,D)
        r = r.expand_as(x)

    lo = x0 - r
    hi = x0 + r
    return torch.max(lo, torch.min(x, hi))


class Annealing:
    def __init__(self, hand_model, switch_possibility=0.5, starting_temperature=18, temperature_decay=0.95, annealing_period=30,
                 step_size=0.005, stepsize_period=50, mu=0.98, device='cpu', init_hand_pose=None):
        """
        Create a optimizer
        
        Use random resampling to update contact point indices
        
        Use RMSProp to update translation, rotation, and joint angles, use step size decay
        
        Use Annealing to accept / reject parameter updates
        
        Parameters
        ----------
        hand_model: hand_model.HandModel
        switch_possibility: float
            possibility to resample each contact point index each step
        starting_temperature: float
        temperature_decay: float
            temperature decay rate and step size decay rate
        annealing_period: int
        step_size: float
        stepsize_period: int
        mu: float
            `1 - decay_rate` of RMSProp
        """

        self.hand_model = hand_model
        self.device = device
        self.switch_possibility = switch_possibility
        self.starting_temperature = torch.tensor(starting_temperature, dtype=torch.float, device=device)
        self.temperature_decay = torch.tensor(temperature_decay, dtype=torch.float, device=device)
        self.annealing_period = torch.tensor(annealing_period, dtype=torch.long, device=device)
        self.step_size = torch.tensor(step_size, dtype=torch.float, device=device)
        self.step_size_period = torch.tensor(stepsize_period, dtype=torch.long, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float, device=device)
        self.step = 0

        self.old_hand_pose = None
        self.old_contact_point_indices = None
        self.old_global_transformation = None
        self.old_global_rotation = None
        self.old_current_status = None
        self.old_contact_points = None
        self.old_grad_hand_pose = None
        self.ema_grad_hand_pose = torch.zeros(self.hand_model.n_dofs + 9, dtype=torch.float, device=device)

        self.init_hand_pose = init_hand_pose

    def try_step(self, fix_wrist=False):
        """
        Try to update translation, rotation, joint angles, and contact point indices
        
        Returns
        -------
        s: torch.Tensor
            current step size
        """

        s = self.step_size * self.temperature_decay ** torch.div(self.step, self.step_size_period, rounding_mode='floor')
        step_size = torch.zeros(*self.hand_model.hand_pose.shape, dtype=torch.float, device=self.device) + s

        self.ema_grad_hand_pose = self.mu * (self.hand_model.hand_pose.grad ** 2).mean(0) + \
            (1 - self.mu) * self.ema_grad_hand_pose

        delta = step_size * self.hand_model.hand_pose.grad / (torch.sqrt(self.ema_grad_hand_pose) + 1e-6)
        # if fix_wrist:
        #     wrist_mask = torch.ones_like(delta)
        #     wrist_mask[:, :3] = 0
        #     delta = delta * wrist_mask
        # hand_pose = self.hand_model.hand_pose - delta

        hand_pose = self.hand_model.hand_pose - delta   # new tensor, part of graph
        if fix_wrist:
            # build rng only for the first K dims, then broadcast to full shape
            rng_list = [0.01, 0.01, 0.01]             # example: clamp translation only
            rng_head = torch.as_tensor(rng_list, device=hand_pose.device, dtype=hand_pose.dtype).view(1, -1)
            K = rng_head.shape[-1]

            rng_full = torch.zeros_like(hand_pose)
            rng_full[:, :K] = rng_head                 # <- out-of-place fill

            x0 = self.init_hand_pose             # detached snapshot (B,D)
            
            lo = x0[:, :K] - rng_head
            hi = x0[:, :K] + rng_head

            firstK = torch.max(lo, torch.min(hand_pose[:, :K], hi))
            hand_pose = torch.cat([firstK, hand_pose[:, K:]], dim=1)

        # now apply to the model WITHOUT creating grads on assignment
        with torch.no_grad():
            self.old_hand_pose = self.hand_model.hand_pose
            self.old_contact_point_indices = self.hand_model.contact_point_indices.clone()
            # (stash other caches as you had)

        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        switch_mask = torch.rand(batch_size, n_contact, dtype=torch.float, device=self.device) < self.switch_possibility
        contact_point_indices = self.hand_model.contact_point_indices.clone()
        contact_point_indices[switch_mask] = torch.randint(self.hand_model.n_contact_candidates, size=[switch_mask.sum()], device=self.device)

        self.old_hand_pose = self.hand_model.hand_pose
        self.old_contact_point_indices = self.hand_model.contact_point_indices
        self.old_global_transformation = self.hand_model.global_translation
        self.old_global_rotation = self.hand_model.global_rotation
        self.old_current_status = self.hand_model.current_status
        self.old_contact_points = self.hand_model.contact_points
        self.old_grad_hand_pose = self.hand_model.hand_pose.grad
        self.hand_model.set_parameters(hand_pose, contact_point_indices)

        self.step += 1

        return s

    def accept_step(self, energy, new_energy):
        """
        Accept / reject updates using annealing
        
        Returns
        -------
        accept: (N,) torch.BoolTensor
        temperature: torch.Tensor
            current temperature
        """

        batch_size = energy.shape[0]
        temperature = self.starting_temperature * self.temperature_decay ** torch.div(self.step, self.annealing_period, rounding_mode='floor')

        alpha = torch.rand(batch_size, dtype=torch.float, device=self.device)
        accept = alpha < torch.exp((energy - new_energy) / temperature)

        with torch.no_grad():
            reject = ~accept
            self.hand_model.hand_pose[reject] = self.old_hand_pose[reject]
            self.hand_model.contact_point_indices[reject] = self.old_contact_point_indices[reject]
            self.hand_model.global_translation[reject] = self.old_global_transformation[reject]
            self.hand_model.global_rotation[reject] = self.old_global_rotation[reject]
            self.hand_model.current_status = self.hand_model.chain.forward_kinematics(self.hand_model.hand_pose[:, 9:])
            self.hand_model.contact_points[reject] = self.old_contact_points[reject]
            self.hand_model.hand_pose.grad[reject] = self.old_grad_hand_pose[reject]

        return accept, temperature

    def zero_grad(self):
        """
        Sets the gradients of translation, rotation, and joint angles to zero
        """
        if self.hand_model.hand_pose.grad is not None:
            self.hand_model.hand_pose.grad.data.zero_()


class Adam:
    def __init__(self, hand_model, switch_possibility=0.5, starting_temperature=18, temperature_decay=0.95,
                 annealing_period=30, step_size=0.005, stepsize_period=50, mu=0.98, device='cpu', init_hand_pose=None):
        """
        使用带有随机接触点重采样的 Adam 优化器

        参数与 Annealing 保持一致，便于在主程序中互换
        """
        self.hand_model = hand_model
        self.device = device
        self.switch_possibility = switch_possibility
        self.step_size = torch.tensor(step_size, dtype=torch.float, device=device)
        self.temperature_decay = torch.tensor(temperature_decay, dtype=torch.float, device=device)
        self.step_size_period = torch.tensor(stepsize_period, dtype=torch.long, device=device)
        # 兼容接口所需参数（在 Adam 中不直接使用，但保留用于统一配置）
        self.starting_temperature = torch.tensor(starting_temperature, dtype=torch.float, device=device)
        self.annealing_period = torch.tensor(annealing_period, dtype=torch.long, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float, device=device)

        self.beta1 = torch.tensor(0.9, dtype=torch.float, device=device)
        self.beta2 = torch.tensor(0.999, dtype=torch.float, device=device)
        self.eps = torch.tensor(1e-8, dtype=torch.float, device=device)

        self.m = torch.zeros_like(self.hand_model.hand_pose, dtype=torch.float, device=device)
        self.v = torch.zeros_like(self.hand_model.hand_pose, dtype=torch.float, device=device)
        self.beta1_pow = torch.tensor(1.0, dtype=torch.float, device=device)
        self.beta2_pow = torch.tensor(1.0, dtype=torch.float, device=device)

        self.init_hand_pose = init_hand_pose
        self.step = 0
        self.current_step_size = self.step_size.clone()

    def _compute_step_size(self):
        decay_step = torch.div(self.step, self.step_size_period, rounding_mode='floor')
        lr = self.step_size * (self.temperature_decay ** decay_step)
        return lr

    def try_step(self, fix_wrist=False):
        """
        使用 Adam 更新手部姿态，并随机重采样接触点
        """
        lr = self._compute_step_size()
        grad = self.hand_model.hand_pose.grad

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        self.beta1_pow = self.beta1_pow * self.beta1
        self.beta2_pow = self.beta2_pow * self.beta2

        m_hat = self.m / (1 - self.beta1_pow)
        v_hat = self.v / (1 - self.beta2_pow)

        update = lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        hand_pose = self.hand_model.hand_pose - update

        if fix_wrist:
            rng_list = [0.01, 0.01, 0.01]
            rng_head = torch.as_tensor(rng_list, device=hand_pose.device, dtype=hand_pose.dtype).view(1, -1)
            K = rng_head.shape[-1]

            rng_full = torch.zeros_like(hand_pose)
            rng_full[:, :K] = rng_head

            x0 = self.init_hand_pose
            lo = x0[:, :K] - rng_head
            hi = x0[:, :K] + rng_head
            firstK = torch.max(lo, torch.min(hand_pose[:, :K], hi))
            hand_pose = torch.cat([firstK, hand_pose[:, K:]], dim=1)

        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        switch_mask = torch.rand(batch_size, n_contact, dtype=torch.float, device=self.device) < self.switch_possibility
        contact_point_indices = self.hand_model.contact_point_indices.clone()
        if switch_mask.any():
            contact_point_indices[switch_mask] = torch.randint(
                self.hand_model.n_contact_candidates,
                size=[switch_mask.sum()],
                device=self.device
            )

        self.hand_model.set_parameters(hand_pose, contact_point_indices)

        self.step += 1
        self.current_step_size = lr

        return lr

    def accept_step(self, energy, new_energy):
        """
        Adam 始终接受更新，保持与 Annealing 相同的接口
        """
        accept = torch.ones_like(energy, dtype=torch.bool, device=energy.device)
        return accept, self.current_step_size

    def zero_grad(self):
        if self.hand_model.hand_pose.grad is not None:
            self.hand_model.hand_pose.grad.data.zero_()
