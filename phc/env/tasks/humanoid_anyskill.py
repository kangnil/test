# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import time
import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as sRot
from phc.utils.flags import flags
import open_clip
import os
import yaml
import numpy as np
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
        self.text_yaml = cfg["env"].get("text_yaml", None)
        self.delta = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._similarity = torch.zeros([self.num_envs], device=self.device, dtype=torch.float32)
        self._punish_counter = torch.zeros([self.num_envs], device=self.device, dtype=torch.int)
        self.clip_features = []
        self.mlip_encoder = FeatureExtractor()

        self.RENDER = True
        self.headless = headless
        self.motionclip_features = []
        self._text_latents = torch.zeros((self.num_envs, 512), dtype=torch.float32,
                                         device=self.device)
        # batch_shape = self.experience_buffer.obs_base_shape
        # self._latent_reset_steps = torch.zeros(batch_shape[-1], dtype=torch.int32, device=self.ppo_device)
        #
        # texts, texts_weights = load_texts(self.text_file)
        # self.text_features = self.mlip_encoder.encode_texts(texts)
        # self.text_weights = torch.tensor(texts_weights, device=self.device)
        # self._text_latents = torch.zeros((batch_shape[-1], 512), dtype=torch.float32,
        #                                  device=self.ppo_device)
        # self._latent_text_idx = torch.zeros((batch_shape[-1],), dtype=torch.long, device=self.ppo_device)
        #

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 512
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def post_physics_step(self):
        super().post_physics_step()

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
        self.cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0, 0] + 1.0,
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


    def render_img(self, sync_frame_time=False):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[:, 0:3] if self.RENDER else self._humanoid_root_states[:, 0:3].cpu().numpy()
        # char_root_rot = self._humanoid_root_states[:, 3:7].cpu().numpy()
        self._cam_prev_char_pos[:] = char_root_pos

        start = time.time()
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

        start = time.time()
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



    def _create_envs(self, num_envs, spacing, num_per_row):
        self.camera_handles = []
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 224
        self.camera_props.height = 224
        self.camera_props.enable_tensors = True
        self.torch_rgba_tensor = torch.zeros([self.num_envs, 224, 224, 3], device=self.device, dtype=torch.float32)

        super()._create_envs(num_envs, spacing, num_per_row)
        return



    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        # set camera handles
        # set 1024 cameras in the same location?????
        camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
        self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(1.2, 1.3, 0.5), gymapi.Vec3(-0.5, 0.7, -0.5))
        self.camera_handles.append(camera_handle)
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
        if len(env_ids) >0:
            obs = self._text_latents[env_ids]
        else:
            obs = self._text_latents
        return obs


    def _compute_reset(self):

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces,
                                                                           self._contact_body_ids,
                                                                           self._rigid_body_pos,
                                                                           self.max_episode_length,
                                                                           self._enable_early_termination,
                                                                           self._termination_heights,
                                                                           self._punish_counter)

        return

    def _compute_reward(self, actions):
        # if self.headless == False:
        #     images = self.render_img()
        # else:
        #     # print("apply the headless mode")
        #     images = self.render_headless()
        #
        # image_features = self.mlip_encoder.encode_images(images)
        #
        # state_embeds = self._rigid_body_state_reshaped
        #
        # # print("we have render")
        # self.clip_features.append(image_features.data.cpu().numpy())
        # self.motionclip_features.append(state_embeds.data.cpu().numpy())
        # image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)





        # curr_rewards, delta, similarity = self.compute_anyskill_reward(image_features_norm, self._text_latents,
        #                                                                      self._latent_text_idxnorm) ??????
        # self.rew_buf[:] = self.reward_raw = curr_rewards?????
        # self.reward_raw = self.reward_raw[:, None]

        return

    def compute_anyskill_reward(self, img_features_norm, text_features_norm, corresponding_id):
        similarity = torch.einsum('ij,ij->i', img_features_norm, text_features_norm[corresponding_id])

        # similarity_m = (100.0 * torch.matmul(img_features_norm, text_features_norm.permute(1, 0))).squeeze()
        # rows = torch.arange(corresponding_id.size(0))
        # similarity_raw = similarity_m[rows, corresponding_id]
        if self.RENDER:
            clip_reward_w = 800
        else:
            clip_reward_w = 3600
            # clip_reward_w = 30000
        self.delta = similarity - self._similarity
        punish_mask = self.delta < 0
        self._punish_counter[punish_mask] += 1

        # # delta
        # # print("clip_reward_w: {}".format(clip_reward_w))
        # clip_reward = clip_reward_w * delta

        # # value
        clip_reward = 0.8 * similarity

        # # global
        # # b = a humanoid
        # # l = kneel
        # alpha = torch.tensor(0.5)
        # # similarity.unsqueeze(1)
        # s = img_features_norm
        # g = text_features_norm[corresponding_id]
        # projL_s = torch.nn.functional.normalize(s, p=2, dim=0)
        # term1 = alpha * projL_s
        # term2 = (1 - alpha) * s - g
        # clip_reward = 1 - 0.5 * torch.norm(term1 + term2, p=2, dim=1)**2

        # # CLIP socre
        # mask = similarity < 0
        # print((mask==True).sum())
        # similarity[mask] = 0
        # clip_score = 2.5 * similarity

        # # RCLIP score
        # clip_reward =  clip_reward - E(clip_score)
        # self._exp_sim()

        self._similarity = similarity
        return clip_reward, self.delta, similarity

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

def load_texts(text_file):
    ext = os.path.splitext(text_file)[1]
    assert ext == ".yaml"
    weights = []
    texts = []
    with open(os.path.join(os.getcwd(), text_file), 'r') as f:
        text_config = yaml.load(f, Loader=yaml.SafeLoader)

    text_list = text_config['texts']
    for text_entry in text_list:
        curr_text = text_entry['text']
        curr_weight = text_entry['weight']
        assert(curr_weight >= 0)
        weights.append(curr_weight)
        texts.append(curr_text)
    return texts, weights


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length,
                           enable_early_termination, termination_heights, _punish_counter):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()

        masked_contact_buf[:, contact_body_ids, :] = 0
        force_threshold = 50
        fall_contact = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        mlip_mask = _punish_counter > 8
        if mlip_mask.shape[0]<=16:
            print("mlip_mask", mlip_mask)

        retain_probability = 0.8
        random_tensor = torch.rand_like(mlip_mask, dtype=torch.float)
        retain_mask = random_tensor < retain_probability
        mlip_mask = mlip_mask * retain_mask.int()

        has_fallen = torch.logical_or(fall_contact, fall_height)
        has_fallen = torch.logical_or(has_fallen, mlip_mask)

        has_failed = has_fallen
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

