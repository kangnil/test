# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import random
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
from PIL import Image
from src.t2v_metrics import t2v_metrics

class HumanoidAnyskill(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        rendering_out = os.path.join("output", "renderings", "clip_similarity")
        os.makedirs(rendering_out, exist_ok=True)
        self.curr_stpes = 0
        self._render_image_path = rendering_out
        self.RENDER = True
        self.SAVE_RENDER = True
        self.SAVE_O3D_RENDER = False
        self._enable_task_obs = False
        self.viewer_follow_root = False
        if self.RENDER:
            self._init_camera()

        if self.num_envs>256:
            self.clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')  # our recommended scoring model

        self.text_command = cfg["env"].get("text_command", "a person is running")

        # self._speed_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_speed = 0.2 * torch.ones([self.num_envs], device=self.device, dtype=torch.float)

        return

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # change the setting to multi envs
        self._cam_prev_char_pos = self._humanoid_root_states[:, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0, 0] + 1.5,
                              self._cam_prev_char_pos[0, 1] + 1.5,
                              self._cam_prev_char_pos[0, 2])
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0, 0],
                                 self._cam_prev_char_pos[0, 1],
                                 self._cam_prev_char_pos[0, 2])
        if self.viewer is not None and self.viewer_follow_root:
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)



    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[:, 0:3].cpu().numpy()
        self._cam_prev_char_pos = char_root_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0, 0], char_root_pos[0, 1], char_root_pos[0, 2])
        new_cam_pos = gymapi.Vec3(char_root_pos[0, 0] + 1.5,
                                  char_root_pos[0, 1] + 1.5,
                                  char_root_pos[0, 2])
        if self.viewer is not None and self.viewer_follow_root:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)


    def render_img(self, angle = 45):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[:, 0:3]

        root_rot = self._humanoid_root_states[:, 3:7]
        heading = torch_utils.calc_heading(root_rot)
        cam_heading = heading + torch.deg2rad(torch.tensor([angle])).to(self.device)
        delta_x =  torch.cos(cam_heading) * 1.5
        delta_y = torch.sin(cam_heading) * 1.5
        self._cam_prev_char_pos[:] = char_root_pos

        if hasattr(self, 'viewer_o3d') and self.viewer_o3d:
            # Set the camera parameters to view the front of the humanoid
            view_ctl = self.o3d_vis.get_view_control()

            # Define camera parameters for front view
            humanoid_center = np.array([char_root_pos[0, 0], char_root_pos[0, 1], char_root_pos[0, 2]])

            camera_position = np.array([char_root_pos[0, 0]+ delta_x[0] ,
                              char_root_pos[0, 1] + delta_y[0],
                              char_root_pos[0, 2]])  # Positioned in front of the humanoid
            up_direction = [0, 0, 1]  # Y-axis as up direction
            view_ctl.set_lookat(humanoid_center)
            view_ctl.set_front((camera_position - humanoid_center) / np.linalg.norm(camera_position - humanoid_center))
            view_ctl.set_up(up_direction)




        #start = time.time()
        for env_id in range(self.num_envs):

            target = gymapi.Vec3(char_root_pos[env_id, 0], char_root_pos[env_id, 1], char_root_pos[env_id, 2])
            pos = gymapi.Vec3(char_root_pos[env_id, 0] + delta_x[env_id],
                              char_root_pos[env_id, 1] + delta_y[env_id],
                              char_root_pos[env_id, 2] )

            self.gym.set_camera_location(self.camera_handles[env_id], self.envs[env_id], pos, target)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for env_id in range(self.num_envs):
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id], self.camera_handles[env_id],
                                                                      gymapi.IMAGE_COLOR)
            self.torch_rgba_tensor[env_id] = gymtorch.wrap_tensor(camera_rgba_tensor)[:, :, :].float()  # [224,224,3] -> IM -> [env,224,224,3]

        # print("time of render {} frames' image: {}".format(env_id, (time.time() - start)))
        self.gym.end_access_image_tensors(self.sim)

        images = self.torch_rgba_tensor[:, :, :, 0:3]
        return images



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
    def _create_envs(self, num_envs, spacing, num_per_row):
        # if (not self.headless):
        #     self._marker_handles = []
        #     self._load_marker_asset()
        self.camera_handles = []
        self.camera_props = gymapi.CameraProperties()
        self.resolution = 224
        self.camera_props.width = self.resolution
        self.camera_props.height = self.resolution
        self.camera_props.enable_tensors = True
        self.torch_rgba_tensor = torch.zeros([self.num_envs, self.resolution, self.resolution, 4], device=self.device, dtype=torch.float32)

        super()._create_envs(num_envs, spacing, num_per_row)
        return


    def clip(self,image_path, text):
        print(image_path)
        image = Image.open(image_path)
        inputs = self.processor(text, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        for i, t in enumerate(text):
            print(i, t, logits_per_image[0][i].item())
        return

    def save_o3d(self, angle="0"):
        temp_image_path = f"{self.curr_image_folder_name}/mesh_{self.curr_stpes}_{angle}.png"
        self.o3d_vis.capture_screen_image(temp_image_path, do_render=True)
        # Resize the saved image to 224x224 and save as PNG
        with Image.open(temp_image_path) as img:
            width, height = img.size
            new_width, new_height = 400, 400

            # Calculate coordinates for cropping the center
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2

            # Crop the center of the image
            img = img.crop((left, top, right, bottom))
            img = img.resize((224, 224))
            img.save(temp_image_path, format="PNG")
        return

    def save_pil(self, images, angle="0"):
        image_data_np = images[0].numpy().astype('uint8')
        # Convert numpy array to a PIL Image
        image = Image.fromarray(image_data_np)
        image.save(f"{self.curr_image_folder_name}/{self.curr_stpes}_{angle}.png")

        return

    # def _draw_task(self):
    #     self._update_marker()
    #     return
    # def _update_marker(self):
    #     if flags.show_traj:
    #
    #         motion_times = (self.progress_buf + 1) * self.dt + self._motion_start_times #+ self._motion_start_times_offset  # + 1 for target.
    #         motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times,
    #                                                           self._global_offset)
    #         root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
    #             motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], \
    #             motion_res["root_ang_vel"], motion_res["dof_vel"], \
    #                 motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res[
    #                 "rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
    #
    #     #     self._marker_pos[:] = ref_rb_pos
    #     #     # self._marker_rotation[..., self._track_bodies_id, :] = ref_rb_rot[..., self._track_bodies_id, :]
    #     #
    #     #     ## Only update the tracking points.
    #     #     if flags.real_traj:
    #     #         self._marker_pos[:] = 1000
    #     #
    #     #     self._marker_pos[..., self._track_bodies_id, :] = ref_rb_pos[..., self._track_bodies_id, :]
    #     #
    #     #     if self._occl_training:
    #     #         self._marker_pos[self.random_occlu_idx] = 0
    #     #
    #     # else:
    #     #     self._marker_pos[:] = 1000
    #     #
    #     # # ######### Heading debug #######
    #     # # points = self.init_root_points()
    #     # # base_quat = self._rigid_body_rot[0, 0:1]
    #     # # base_quat = remove_base_rot(base_quat)
    #     # # heading_rot = torch_utils.calc_heading_quat(base_quat)
    #     # # show_points = quat_apply(heading_rot.repeat(1, points.shape[0]).reshape(-1, 4), points) + (self._rigid_body_pos[0, 0:1]).unsqueeze(1)
    #     # # self._marker_pos[:] = show_points[:, :self._marker_pos.shape[1]]
    #     # # ######### Heading debug #######
    #     #
    #     # self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
    #     #                                              gymtorch.unwrap_tensor(self._marker_actor_ids),
    #     #                                              len(self._marker_actor_ids))
    #     return

    # def get_task_obs_size(self):
    #     obs_size = 0
    #     if (self._enable_task_obs):
    #         obs_size = 512
    #
    #     # if (self._add_input_noise):
    #     #     obs_size += 16
    #     #
    #     # if self.obs_v == 2:
    #     #     obs_size *= self.past_track_steps
    #
    #     return obs_size


    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def post_physics_step(self):
        super().post_physics_step()

        # if (humanoid_amp.HACK_OUTPUT_MOTION):
        #     self._hack_output_motion_target()

        return

    def compute_vqascore_reward(self, curr_images, text_command):
        score = self.clip_flant5_score(images=curr_images, texts=text_command)
        return score


    def _compute_reward(self, actions):
        vqascore_rewards  = torch.zeros([self.num_envs], device=self.device, dtype=torch.float32)
        if self.num_envs>256:
            vqascore_rewards = self.compute_vqascore_reward(self.curr_images, self.text_command)
        self.rew_buf[:] = self.reward_raw = vqascore_rewards
        self.reward_raw = self.reward_raw[:, None]

        return


    
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
        if self.SAVE_RENDER or self.SAVE_O3D_RENDER:
            curr_motion_key = self._motion_lib.curr_motion_keys[0]
            curr_motion_id = self._motion_lib._curr_motion_ids[0].item()
            curr_motion = str(curr_motion_id) + "-" + curr_motion_key #+ "-" + datetime.now().strftime( '%Y-%m-%d-%H:%M:%S')
            self.curr_image_folder_name = os.path.join(self._render_image_path , curr_motion)
            os.makedirs(self.curr_image_folder_name, exist_ok=True)



        if self.RENDER:
            angle = random.randint(-90, 90)
            self.curr_images = self.render_img(angle)
            if self.SAVE_RENDER:
                self.save_pil(self.curr_images, angle)
            if self.SAVE_O3D_RENDER:
                self.save_o3d(angle)

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


