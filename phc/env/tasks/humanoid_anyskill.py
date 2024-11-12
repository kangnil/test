# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import random
import torch
import time
import copy
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
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import TexturesUV,TexturesVertex
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.transforms import euler_angles_to_matrix,so3_exp_map,quaternion_to_matrix
# Data structures and functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,

)
from pytorch3d.renderer.mesh.shader import TexturedSoftPhongShader
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

import joblib
from smpl_sim.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


class HumanoidAnyskill(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.viewer_follow_root = False
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.llm_model = cfg["env"].get("llm_model", 'clip-flant5-xxl')
        self.llm_model_device = cfg["env"].get("llm_model_device", 'cuda:0')
        self.llm_model_batchsize = cfg["env"].get("llm_model_batchsize", 32)
        self.text_command = cfg["env"].get("text_command", "a person is running")
        self.opengl_render = cfg["env"].get("opengl_render", False)


        rendering_out = os.path.join("output", "renderings", self.text_command, self.llm_model)
        os.makedirs(rendering_out, exist_ok=True)
        self.curr_stpes = 0
        self._render_image_path = rendering_out
        self.RENDER = False
        self.SAVE_RENDER = False
        self.SAVE_O3D_RENDER = False
        self._enable_task_obs = False
        self.vis_ref=False

        if not self.opengl_render:
            self._init_camera()

        if self.num_envs>16:
            if self.llm_model=='gpt-4o':
                openai_key = "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"
                self.clip_flant5_score = t2v_metrics.get_score_model(model="gpt-4o", device="cuda", openai_key=openai_key, top_logprobs=20)
            elif self.llm_model in ['llava-v1.5-13b', 'llava-v1.5-7b', 'clip-flant5-xxl', 'clip-flant5-xl'] :
                print(self.llm_model, "-----------------------------------------")
                self.clip_flant5_score = t2v_metrics.VQAScore(model=self.llm_model, device=self.llm_model_device)
                #self.clip_flant5_score = t2v_metrics.VQAScore(model=self.llm_model, device='cuda')
            else:
                print("check-----------------------------------------------")

        
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

    def save_pil(self, images, angle):
        image_data_np = images.numpy().astype('uint8')
        # Convert numpy array to a PIL Image
        image = Image.fromarray(image_data_np)
        image.save(f"{self.curr_image_folder_name}/{self.llm_model}_{angle:.2f}.png")

        return

 
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def post_physics_step(self):
        super().post_physics_step()

        # if (humanoid_amp.HACK_OUTPUT_MOTION):
        #     self._hack_output_motion_target()

        return



    def render_opengl_images(self):

        device = self.device

        # Initialize an OpenGL perspective camera.
        # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
        # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
        # Example: Create a custom rotation (e.g., rotate around X axis by 45 degrees)
        # Define rotation angles for each axis in radians (e.g., rotate 45 degrees around X-axis)
        #angles = torch.tensor([90, -90, 0], dtype=torch.float32) * (3.14159265 / 180.0)  # Convert degrees to radians

        # Create rotation matrix from Euler angles
        #R = euler_angles_to_matrix(angles, "XYZ").unsqueeze(0)  # Shape (1, 3, 3)
        R = torch.tensor([[[0,    0,   -1.0],
                           [-1.0, 0,   0],
                           [0,    1.0, 0]]])
        # Define translation for the camera
        root_trans = self._rigid_body_pos[self.num_envs-1, 0, :].cpu()
        T = torch.tensor([[0, 0, 2]], dtype=torch.float32)    # Shape (1, 3)
        R_root_trans = torch.matmul(torch.inverse(R), root_trans)
        T-=R_root_trans
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. Refer to rasterize_meshes.py for explanations of these parameters.
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
        # -z direction.
        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

        # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=TexturedSoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )



        #############################################
        if self.opengl_render:
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:

                if self._has_upright_start:
                    self.pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
                    from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, \
                        SMPLH_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES

                    self.mujoco_2_smpl = [self._body_names_orig.index(q) for q in SMPL_BONE_ORDER_NAMES if
                                          q in self._body_names_orig]
                    self.mesh_parser = copy.deepcopy(self._motion_lib.mesh_parsers[0])
                    self.mesh_parser = self.mesh_parser.cuda()

                    body_quat = self._rigid_body_rot
                    root_trans = self._rigid_body_pos[:, 0, :]

                    if self.vis_ref and len(self.ref_motion_cache['dof_pos']) == self.num_envs:
                        ref_body_quat = self.ref_motion_cache['rb_rot']
                        ref_root_trans = self.ref_motion_cache['root_pos']

                        body_quat = torch.cat([body_quat, ref_body_quat])
                        root_trans = torch.cat([root_trans, ref_root_trans])

                    N = body_quat.shape[0]
                    offset = torch.zeros_like(root_trans)
                    for i in range(N):
                        offset[i] = self.skeleton_trees[i].local_translation[0]
                    root_trans_offset = root_trans - offset.to(device)

                    pose_quat = (sRot.from_quat(body_quat.reshape(-1, 4).numpy()) * self.pre_rot).as_quat().reshape(N,  -1,  4)
                    local_rot = torch.zeros_like(body_quat)
                    for i in range(N):
                        new_sk_state = SkeletonState.from_rotation_and_root_translation(self.skeleton_trees[i],
                                                                                    torch.from_numpy(pose_quat[i:i+1]),
                                                                                    root_trans[i:i+1].cpu(), is_local=False)
                        local_rot[i] = new_sk_state.local_rotation
                    pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(N, -1, 3)
                    pose_aa = torch.from_numpy(pose_aa[:, self.mujoco_2_smpl, :].reshape(N, -1)).cuda()
                else:
                    print("not implemented")
                with torch.no_grad():
                    vertices, joints = self.mesh_parser.get_joints_verts(pose=pose_aa, th_trans=root_trans_offset.cuda())

                vertices = vertices.to(device)
                faces = torch.from_numpy(self.mesh_parser.faces.astype(np.int32)).to(device)  # 13776,3. min 0 max 6889
                img = cv2.cvtColor(cv2.imread("scripts/render/nongrey_male_0540.jpg"), cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).float().to(device) / 255.0  # Convert to float and normalize to [0, 1]
                # create the uv coordinates

                v_uv = np.load("scripts/render/smpl_uv_map.npy")  # Shape (6890, 2)
                v_uv = torch.from_numpy(v_uv).to(device, dtype=torch.float32)  # Shape (1, 6890, 2)

                batch = self.num_envs
                verts_uvs = v_uv.unsqueeze(0).repeat(batch, 1, 1)  # Shape: (1, 6890, 2)
                faces_uvs = faces.unsqueeze(0).repeat(batch, 1, 1)

                # Shape: (1, 13776, 3)
                img = img.unsqueeze(0).repeat(batch, 1, 1, 1)
                textures = TexturesUV(maps=img, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
                textures = textures.to(device)
                smpl_meshes = Meshes(
                    verts=vertices,
                    faces=faces_uvs,
                    textures=textures
                )

                start_time = time.time()
                renderer = renderer.to(device)
                rendered_image = renderer(smpl_meshes).cpu().numpy()[..., 0:3]

                floor_image_path = "scripts/render/floor.jpg"  # Replace with the path to your floor image
                floor_image = cv2.imread(floor_image_path)
                floor_image = cv2.cvtColor(floor_image, cv2.COLOR_BGR2RGB)  # Convert to RGB format for compatibility

                # Resize the floor image to match the rendered image dimensions
                rendered_image_height, rendered_image_width = rendered_image.shape[1:3]
                floor_image_resized = cv2.resize(floor_image, (rendered_image_width, rendered_image_height))/255

                # Step 3: Overlay the rendered image on the floor image
                # If your rendered image has a black or other color background, you may want to create a mask
                # Here we assume that the rendered image has a black background, so we create a mask for non-black pixels

                mask = np.all(rendered_image<1, axis=-1)


                # Combine the images using the mask
                floor_image_expanded = np.expand_dims(floor_image_resized, axis=0)  # Shape: (1, H, W, C)

                # Repeat the image along the batch dimension
                combined_image = np.repeat(floor_image_expanded, batch, axis=0)  # Shape: (batch, H, W, C)

                combined_image[mask] = rendered_image[mask]
                end_time = time.time()
                render_time = end_time - start_time

                print("render time is ", render_time)
                plt.figure(dpi=250)
                plt.imshow(combined_image[self.num_envs-1])
                plt.grid("off");
                plt.axis("off")
                plt.show()
                return combined_image
        return
    def compute_vqascore_reward(self, curr_images, text_command):

        # if self.llm_model in []:
        #     curr_images_convert = []
        #     for i in curr_images:
        #         if i.dtype != torch.uint8:
        #             i = i.to(torch.uint8)
        #         curr_images_convert.append(Image.fromarray(i.numpy()))
        #     #print(type(curr_images_convert))
        #     score = self.clip_flant5_score(images=curr_images_convert, texts=text_command) 
        #     print(score[0])
        if self.llm_model in ['clip-flant5-xl', 'clip-flant5-xxl','llava-v1.5-13b' ,'llava-v1.5-7b']:
            score = self.clip_flant5_score.batch_forward(dataset_image=curr_images, dataset_text=text_command, batch_size=self.llm_model_batchsize) # (n_sample, 4, 1) tensor
            self.save_pil(curr_images[0], score[0].item())
        elif self.llm_model=='gpt-4o':
            score = self.clip_flant5_score(images=curr_images, texts=text_command)
                
            # curr_images = ["/home/kangnil/pulse/output/renderings/clip_similarity/223-0-ACCAD_Male2Running_c3d_C24 - quick sidestep left_poses/0_-9.png"]
            # score = self.clip_flant5_score(images=curr_images, texts=text_command)
        else:
            print("check----------------------------------------------------------")



 
        return score


    def _compute_reward(self, actions):
        if self.num_envs>16:
            vqascore_rewards = self.compute_vqascore_reward(self.curr_images, self.text_command).squeeze(dim=1)
        else:
            vqascore_rewards  = torch.zeros([self.num_envs], device=self.device, dtype=torch.float32)



        self.rew_buf[:] = self.reward_raw = vqascore_rewards.to(self.device)
        self.reward_raw = self.reward_raw[:, None]

        return
 
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


        if self.opengl_render:
            self.curr_images = self.render_opengl_images()
        if self.RENDER:
            angle = random.randint(-90, 90)
            self.curr_images = self.render_img(angle)
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



# @torch.jit.script
# def compute_speed_reward(root_pos, prev_root_pos, root_rot, tar_speed, dt):
#     # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor
#     vel_err_scale = 0.25
#     tangent_err_w = 0.1

#     delta_root_pos = root_pos - prev_root_pos
#     root_vel = delta_root_pos / dt
#     tar_dir_speed = root_vel[..., 0]
#     tangent_speed = root_vel[..., 1]

#     tar_vel_err = tar_speed - tar_dir_speed
#     tangent_vel_err = tangent_speed
#     dir_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err +  tangent_err_w * tangent_vel_err * tangent_vel_err))

#     vel_reward_w = 0.01
#     reward = dir_reward * vel_reward_w

#     return reward


