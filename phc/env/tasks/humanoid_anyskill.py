# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os

import torch
from torchvision import transforms
import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as sRot
from phc.utils.flags import flags
import open_clip

TAR_ACTOR_ID = 1
class FeatureExtractor():
    def __init__(self):
        self.mlip_model, _, self.mlip_preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                                         pretrained='laion2b_s34b_b79k', device="cuda")
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def encode_texts(self, texts):
        texts_token = self.tokenizer(texts).cuda()
        text_features = self.mlip_model.encode_text(texts_token).cuda()
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features_norm

    def encode_images(self, images):
        return self.mlip_model.encode_image(images)

class HumanoidAnyskill(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # self._tar_speed_min = cfg["env"]["tarSpeedMin"]
        # self._tar_speed_max = cfg["env"]["tarSpeedMax"]
        # self._speed_change_steps_min = cfg["env"]["speedChangeStepsMin"]
        # self._speed_change_steps_max = cfg["env"]["speedChangeStepsMax"]
        #
        # self._add_input_noise = cfg["env"].get("addInputNoise", False)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.RENDER = True
        if self.viewer is not None:
            self._init_camera()
        else:
            if self.RENDER:
                self._init_camera_headless()
        self.mlip_encoder = FeatureExtractor()
        self.text_command = cfg["env"].get("text_command", "kick")
        self.text_features_norm = self.mlip_encoder.encode_texts(self.text_command).repeat(self.num_envs, 1)

        self._similarity = torch.zeros([self.num_envs], device=self.device, dtype=torch.float32)
        #self._punish_counter = torch.zeros([self.num_envs], device=self.device, dtype=torch.int)


        # self._speed_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_speed = 0.2 * torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        #
        # self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        # reward_raw_num = 1
        # if self.power_usage_reward:
        #     reward_raw_num += 1
        # if self.power_reward:
        #     reward_raw_num += 1
        #
        # self.reward_raw = torch.zeros((self.num_envs, reward_raw_num)).to(self.device)
        # self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        # self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        # self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        #
        # if (not self.headless):
        #     self._build_marker_state_tensors()

        return
    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # change the setting to multi envs
        self._cam_prev_char_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float32)
        # self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        # cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0, 0] + 3.0,
        #                       self._cam_prev_char_pos[0, 1] - 0.5,
        #                       1.0)
        # cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0, 0] + 3.0,
        #                       self._cam_prev_char_pos[0, 1],
        #                       1.0) # zheng
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0, 0],
                              self._cam_prev_char_pos[0, 1] - 3.0,
                              1.0) # ce
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0, 0],
                                 self._cam_prev_char_pos[0, 1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _init_camera_headless(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float32)
        self.cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0, 0] + 3.0,
                              self._cam_prev_char_pos[0, 1],
                              1.0)
        # cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0, 0],
        #                       self._cam_prev_char_pos[0, 1] - 3.0,
        #                       1.0)
        self.cam_target = gymapi.Vec3(self._cam_prev_char_pos[0, 0],
                                 self._cam_prev_char_pos[0, 1],
                                 1.0)
        # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[:, 0:3].cpu().numpy()
        self._cam_prev_char_pos = char_root_pos

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos[0]

        new_cam_target = gymapi.Vec3(char_root_pos[0, 0], char_root_pos[0, 1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0, 0] + cam_delta[0],
                                  char_root_pos[0, 1] + cam_delta[1],
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)


    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 512

        # if (self._add_input_noise):
        #     obs_size += 16
        #
        # if self.obs_v == 2:
        #     obs_size *= self.past_track_steps
        
        return obs_size


    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def post_physics_step(self):
        super().post_physics_step()

        # if (humanoid_amp.HACK_OUTPUT_MOTION):
        #     self._hack_output_motion_target()

        return
    

    def _create_envs(self, num_envs, spacing, num_per_row):
        # if (not self.headless):
        #     self._marker_handles = []
        #     self._load_marker_asset()
        self.camera_handles = []
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 224
        self.camera_props.height = 224
        self.camera_props.enable_tensors = True
        self.torch_rgba_tensor = torch.zeros([self.num_envs, 224, 224, 3], device=self.device, dtype=torch.float32)

        super()._create_envs(num_envs, spacing, num_per_row)
        return


    def _build_env(self, env_id, env_ptr, humanoid_asset):

        # set camera handles
        # set 1024 cameras in the same location?????
        camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
        self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(1.2, 1.3, 0.5), gymapi.Vec3(-0.5, 0.7, -0.5))
        self.camera_handles.append(camera_handle)

        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        # if (not self.headless):
        #     self._build_marker(env_id, env_ptr)

        return


    def render_img(self, sync_frame_time=False):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[:, 0:3] if self.RENDER else self._humanoid_root_states[:, 0:3].cpu().numpy()
        # char_root_rot = self._humanoid_root_states[:, 3:7].cpu().numpy()
        self._cam_prev_char_pos[:] = char_root_pos

        #start = time.time()
        for env_id in range(self.num_envs):
            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = torch.tensor([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z], device=self.device)
            cam_delta = cam_pos - self._cam_prev_char_pos[env_id]

            target = gymapi.Vec3(char_root_pos[env_id, 0], char_root_pos[env_id, 1], 1.0)
            pos = gymapi.Vec3(char_root_pos[env_id, 0] + cam_delta[0],
                              char_root_pos[env_id, 1] + cam_delta[1],
                              cam_pos[2])

            self.gym.viewer_camera_look_at(self.viewer, None, pos, target)
            pos_nearer = gymapi.Vec3(pos.x + 1.2, pos.y + 1.2, pos.z)
            self.gym.set_camera_location(self.camera_handles[env_id], self.envs[env_id], pos_nearer, target)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for env_id in range(self.num_envs):
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id], self.camera_handles[env_id],
                                                                      gymapi.IMAGE_COLOR)
            self.torch_rgba_tensor[env_id] = gymtorch.wrap_tensor(camera_rgba_tensor)[:, :, :3].float()  # [224,224,3] -> IM -> [env,224,224,3]
        # print("time of render {} frames' image: {}".format(env_id, (time.time() - start)))
        self.gym.end_access_image_tensors(self.sim)

        return self.torch_rgba_tensor.permute(0, 3, 1, 2)

    def render_headless(self, sync_frame_time=False):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[:, 0:3]
        # char_root_rot = self._humanoid_root_states[:, 3:7].cpu().numpy()
        self._cam_prev_char_pos[:] = char_root_pos

        #start = time.time()
        for env_id in range(self.num_envs):
            cam_pos = torch.tensor([self.cam_pos.x, self.cam_pos.y, self.cam_pos.z], device=self.device)
            cam_delta = cam_pos - self._cam_prev_char_pos[env_id]

            target = gymapi.Vec3(char_root_pos[env_id, 0], char_root_pos[env_id, 1], 1.0)
            pos = gymapi.Vec3(char_root_pos[env_id, 0] + cam_delta[0],
                              char_root_pos[env_id, 1] + cam_delta[1],
                              cam_pos[2])

            pos_nearer = gymapi.Vec3(pos.x + 1.2, pos.y + 1.2, pos.z)
            self.gym.set_camera_location(self.camera_handles[env_id], self.envs[env_id], pos_nearer, target)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # todo: THE CAMERA VIEW CHANGE STEP BY STEP

        for env_id in range(self.num_envs):
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id],
                                                                      self.camera_handles[env_id],
                                                                      gymapi.IMAGE_COLOR)
            self.torch_rgba_tensor[env_id] = gymtorch.wrap_tensor(camera_rgba_tensor)[:, :,
                                             :3].float()  # [224,224,3] -> IM -> [env,224,224,3]
        # print("time of render {} frames' image: {}".format(env_id, (time.time() - start)))
        self.gym.end_access_image_tensors(self.sim)

        return self.torch_rgba_tensor.permute(0, 3, 1, 2)

    # def _reset_env_tensors(self, env_ids):
    #     super()._reset_env_tensors(env_ids)
    #     self._punish_counter[env_ids] = 0
    #     if 0 in env_ids:
    #         print("punish counter 0 reset")
    #     return


    # def _update_task(self):
    #     reset_task_mask = self.progress_buf >= self._speed_change_steps
    #     rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
    #
    #
    #     if len(rest_env_ids) > 0:
    #         self._reset_task(rest_env_ids)
    #     return

    # def _reset_task(self, env_ids):
    #     n = len(env_ids)
    #
    #     tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
    #     change_steps = torch.randint(low=self._speed_change_steps_min, high=self._speed_change_steps_max,
    #                                  size=(n,), device=self.device, dtype=torch.int64)
    #
    #     self._tar_speed[env_ids] = tar_speed
    #     self._speed_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
    #     return
    
    # def _compute_flip_task_obs(self, normal_task_obs, env_ids):
    #     B, D = normal_task_obs.shape
    #     flip_task_obs = normal_task_obs.clone()
    #     flip_task_obs[:, 1] = -flip_task_obs[:, 1]
    #
    #     return flip_task_obs
    
    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            obs = self.text_features_norm
        else:
            # root_states = self._humanoid_root_states[env_ids]
            # tar_speed = self._tar_speed[env_ids]
            obs = self.text_features_norm[env_ids]
        

        #compute_anyskill_observations(self.text_command)

        # if self._add_input_noise:
        #     obs = torch.cat([obs, torch.randn((obs.shape[0], 16)).to(obs) * 0.1], dim=-1)

        return obs

    def compute_anyskill_reward(self ):
        similarity = torch.einsum('ij,ij->i',  self.image_features_norm,  self.text_features_norm)
        self.delta = similarity - self._similarity
        #punish_mask = self.delta < 0
        #self._punish_counter[punish_mask] += 1

        # # value
        clip_reward = similarity

        self._similarity = similarity
        return clip_reward, self.delta, similarity

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        
        # if False:
        if flags.test:
            root_pos = self._humanoid_root_states[..., 0:3]
            delta_root_pos = root_pos - self._prev_root_pos
            root_vel = delta_root_pos / self.dt
            tar_dir_speed = root_vel[..., 0]
            # print(self._tar_speed, tar_dir_speed)
        anyskill_rewards, delta, similarity = self.compute_anyskill_reward()
        aux_reward = compute_speed_reward(root_pos, self._prev_root_pos,  root_rot, self._tar_speed, self.dt)
        self.rew_buf[:] = self.reward_raw = aux_reward + anyskill_rewards
        self.reward_raw = self.reward_raw[:, None]

        # # if True:
        # if self.power_reward:
        #     power_all = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel))
        #     power = power_all.sum(dim=-1)
        #     power_reward = -self.power_coefficient * power
        #     power_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped.
        #
        #     self.rew_buf[:] += power_reward
        #     self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)
        #
        # # if True:
        # if self.power_usage_reward:
        #     power_all = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel))
        #     power_all = power_all.reshape(-1, 23, 3)
        #     left_power = power_all[:, self.left_indexes].reshape(self.num_envs, -1).sum(dim = -1)
        #     right_power = power_all[:, self.right_indexes].reshape(self.num_envs, -1).sum(dim = -1)
        #     self.power_acc[:, 0] += left_power
        #     self.power_acc[:, 1] += right_power
        #     power_usage_reward = self.power_acc/(self.progress_buf + 1)[:, None]
        #     # print((power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs())
        #     power_usage_reward = - self.power_usage_coefficient * (power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs()
        #     power_usage_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped. on the ground to balance.
        #
        #     self.rew_buf[:] += power_usage_reward
        #     self.reward_raw = torch.cat([self.reward_raw, power_usage_reward[:, None]], dim=-1)
        #

        return

    # def _draw_task(self):
    #     self._update_marker()
    #     return
    
    # def _reset_ref_state_init(self, env_ids):
    #     super()._reset_ref_state_init(env_ids)
    #     self.power_acc[env_ids] = 0
    
    # def _sample_ref_state(self, env_ids):
    #     motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel  = super()._sample_ref_state(env_ids)
    #
    #     # ZL Hack: Forcing to always be facing the x-direction.
    #     if not self._has_upright_start:
    #         heading_rot_inv = torch_utils.calc_heading_quat_inv(humanoid_amp.remove_base_rot(root_rot))
    #     else:
    #         heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)
    #
    #
    #
    #     heading_rot_inv_repeat = heading_rot_inv[:, None].repeat(1, len(self._body_names), 1)
    #     root_rot = quat_mul(heading_rot_inv, root_rot).clone()
    #     rb_pos = quat_apply(heading_rot_inv_repeat, rb_pos - root_pos[:, None, :]).clone() + root_pos[:, None, :]
    #     rb_rot = quat_mul(heading_rot_inv_repeat, rb_rot).clone()
    #     root_ang_vel = quat_apply(heading_rot_inv, root_ang_vel).clone()
    #     root_vel = quat_apply(heading_rot_inv, root_vel).clone()
    #     body_vel = quat_apply(heading_rot_inv_repeat, body_vel).clone()
    #
    #     return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel
    #


class HumanoidAnyskillZ(HumanoidAnyskill):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.initialize_z_models()
        return
    
    def step(self, actions):
        if self.RENDER:
            if self.cfg.headless == False:
                images = self.render_img()
                # transform = transforms.ToPILImage()
                # pil_image = transform(images[0])
                # os.makedirs("output/renderings/rendered/", exist_ok=True)
                # pil_image.save(f"output/renderings/rendered/rendered.jpg")

            else:
                images = self.render_headless()
                #transform = transforms.ToPILImage()
                #pil_image = transform(images[0])
                #os.makedirs("output/renderings/rendered/", exist_ok=True)
                #pil_image.save(f"output/renderings/rendered/rendered_headless.jpg")
            image_features = self.mlip_encoder.encode_images(images)
            self.image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        self.step_z(actions)
        return
     
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()
        
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################



@torch.jit.script
def compute_speed_reward(root_pos, prev_root_pos, root_rot, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    vel_err_scale = 0.25
    tangent_err_w = 0.1

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = root_vel[..., 0]
    tangent_speed = root_vel[..., 1]

    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err +  tangent_err_w * tangent_vel_err * tangent_vel_err))

    vel_reward_w = 0.01
    reward = dir_reward * vel_reward_w

    return reward


