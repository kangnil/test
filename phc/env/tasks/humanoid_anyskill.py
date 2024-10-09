# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as sRot
from phc.utils.flags import flags

TAR_ACTOR_ID = 1


class HumanoidAnyskill(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):


        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)


        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        reward_raw_num = 1
        if self.power_usage_reward:
            reward_raw_num += 1
        if self.power_reward:
            reward_raw_num += 1

        self.reward_raw = torch.zeros((self.num_envs, reward_raw_num)).to(self.device)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2)).to(self.device)

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 2
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def post_physics_step(self):
        super().post_physics_step()

        return



    def _create_envs(self, num_envs, spacing, num_per_row):

        super()._create_envs(num_envs, spacing, num_per_row)
        return



    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)


        return





    # def _update_task(self):
    #     reset_task_mask = self.progress_buf >= self._speed_change_steps
    #     rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
    #
    #     if len(rest_env_ids) > 0:
    #         self._reset_task(rest_env_ids)
    #     return
    #
    # def _reset_task(self, env_ids):
    #
    #     return

    def _compute_flip_task_obs(self, normal_task_obs, env_ids):
        B, D = normal_task_obs.shape
        flip_task_obs = normal_task_obs.clone()
        flip_task_obs[:, 1] = -flip_task_obs[:, 1]

        return flip_task_obs

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states

        else:
            root_states = self._humanoid_root_states[env_ids]


        obs = compute_anyskill_observations(root_states)


        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]



        self.rew_buf[:] = self.reward_raw = compute_anyskill_reward(root_pos, self._prev_root_pos, root_rot,  self.dt)
        self.reward_raw = self.reward_raw[:, None]

        # if True:
        if self.power_reward:
            power_all = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel))
            power = power_all.sum(dim=-1)
            power_reward = -self.power_coefficient * power
            power_reward[
                self.progress_buf <= 3] = 0  # First 3 frame power reward should not be counted. since they could be dropped.

            self.rew_buf[:] += power_reward
            self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)

        # if True:
        if self.power_usage_reward:
            power_all = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel))
            power_all = power_all.reshape(-1, 23, 3)
            left_power = power_all[:, self.left_indexes].reshape(self.num_envs, -1).sum(dim=-1)
            right_power = power_all[:, self.right_indexes].reshape(self.num_envs, -1).sum(dim=-1)
            self.power_acc[:, 0] += left_power
            self.power_acc[:, 1] += right_power
            power_usage_reward = self.power_acc / (self.progress_buf + 1)[:, None]
            # print((power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs())
            power_usage_reward = - self.power_usage_coefficient * (
                        power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs()
            power_usage_reward[
                self.progress_buf <= 3] = 0  # First 3 frame power reward should not be counted. since they could be dropped. on the ground to balance.

            self.rew_buf[:] += power_usage_reward
            self.reward_raw = torch.cat([self.reward_raw, power_usage_reward[:, None]], dim=-1)

        return



    def _reset_ref_state_init(self, env_ids):
        super()._reset_ref_state_init(env_ids)
        self.power_acc[env_ids] = 0

    def _sample_ref_state(self, env_ids):
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel = super()._sample_ref_state(
            env_ids)

        # ZL Hack: Forcing to always be facing the x-direction.
        if not self._has_upright_start:
            heading_rot_inv = torch_utils.calc_heading_quat_inv(humanoid_amp.remove_base_rot(root_rot))
        else:
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

        heading_rot_inv_repeat = heading_rot_inv[:, None].repeat(1, len(self._body_names), 1)
        root_rot = quat_mul(heading_rot_inv, root_rot).clone()
        rb_pos = quat_apply(heading_rot_inv_repeat, rb_pos - root_pos[:, None, :]).clone() + root_pos[:, None, :]
        rb_rot = quat_mul(heading_rot_inv_repeat, rb_rot).clone()
        root_ang_vel = quat_apply(heading_rot_inv, root_ang_vel).clone()
        root_vel = quat_apply(heading_rot_inv, root_vel).clone()
        body_vel = quat_apply(heading_rot_inv_repeat, body_vel).clone()

        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel



class HumanoidAnyskillZ(HumanoidAnyskill):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type,
                         device_id=device_id, headless=headless)
        self.initialize_z_models()
        return

    def step(self, actions):
        self.step_z(actions)
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def compute_anyskill_observations(root_states):
    # type: (Tensor) -> Tensor
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.zeros_like(root_states[..., 0:3])
    tar_dir3d[..., 0] = 1
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_dir = torch_utils.my_quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]

    obs =  local_tar_dir

    return obs


#@torch.jit.script
def compute_anyskill_reward(root_pos, prev_root_pos, root_rot, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    vel_err_scale = 0.25
    tangent_err_w = 0.1

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = root_vel[..., 0]
    tangent_speed = root_vel[..., 1]


    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(
        -vel_err_scale * ( tangent_err_w * tangent_vel_err * tangent_vel_err))

    reward = dir_reward

    return reward