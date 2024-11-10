import os
from random import triangular
import time
import numpy
import torch
import matplotlib.pyplot as plt
from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.transforms import euler_angles_to_matrix
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
import numpy as np
# add path for demo utils functions
import sys
import os
sys.path.append(os.path.abspath(''))
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




def main():
    device = torch.device('cuda', index=1) if torch.cuda.is_available() else torch.device('cpu')

    # Initialize an OpenGL perspective camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
    # Example: Create a custom rotation (e.g., rotate around X axis by 45 degrees)
    # Define rotation angles for each axis in radians (e.g., rotate 45 degrees around X-axis)
    angles = torch.tensor([90, -90, 0], dtype=torch.float32) * (3.14159265 / 180.0)  # Convert degrees to radians

    # Create rotation matrix from Euler angles
    R = euler_angles_to_matrix(angles, "XYZ").unsqueeze(0)  # Shape (1, 3, 3)

    # Define translation for the camera
    T = torch.tensor([[0, -1, 2]], dtype=torch.float32)  # Shape (1, 3)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to rasterize_meshes.py for explanations of these parameters.
    raster_settings = RasterizationSettings(
        image_size=224,
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





    pkl_dir = "output/states/pulse_motionfile_standing_text_a_person_is_bending_upper_body_forward_xl-2024-11-05-17:16:34.pkl"
    mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
                          'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
                          'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    Name = pkl_dir.split("/")[-1].split(".")[0]
    pkl_data = joblib.load(pkl_dir)
    data_dir = "data/smpl"
    mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]

    # Initialize SMPL parsers based on gender
    smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
    smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
    smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")

    # Select the appropriate SMPL parser based on data gender
    data_seq = pkl_data['0_0']
    pose_quat, trans = data_seq['body_quat'].numpy()[::2], data_seq['trans'].numpy()[::2]
    skeleton_tree = SkeletonTree.from_dict(data_seq['skeleton_tree'])
    offset = skeleton_tree.local_translation[0]
    root_trans_offset = trans - offset.numpy()
    gender, beta = data_seq['betas'][0], data_seq['betas'][1:]

    if gender == 0:
        smpl_parser = smpl_parser_n
    elif gender == 1:
        smpl_parser = smpl_parser_m
    else:
        smpl_parser = smpl_parser_f

    sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat),
                                                                torch.from_numpy(trans), is_local=True)

    global_rot = sk_state.global_rotation
    B, J, N = global_rot.shape
    pose_quat = (sRot.from_quat(global_rot.reshape(-1, 4).numpy()) * sRot.from_quat(
        [0.5, 0.5, 0.5, 0.5])).as_quat().reshape(B, -1, 4)
    B_down = pose_quat.shape[0]
    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat),
                                                                    torch.from_numpy(trans), is_local=False)
    local_rot = new_sk_state.local_rotation
    pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(B_down, -1, 3)
    pose_aa = pose_aa[:, mujoco_2_smpl, :].reshape(B_down, -1)
    root_trans_offset[..., :2] = root_trans_offset[..., :2] - root_trans_offset[0:1, :2]
    with torch.no_grad():
        vertices, joints = smpl_parser.get_joints_verts(pose=torch.from_numpy(pose_aa),
                                                        th_trans=torch.from_numpy(root_trans_offset),
                                                        th_betas=torch.from_numpy(beta[None,]))

    #99,6890,3
    faces = torch.from_numpy(smpl_parser.faces.astype(np.int32)).to(device) #13776,3. min 0 max 6889
    vertices = vertices[0].to(device)

    img = cv2.cvtColor(cv2.imread("scripts/render/nongrey_male_0540.jpg"), cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float().to(device) / 255.0  # Convert to float and normalize to [0, 1]
    # create the uv coordinates

    v_uv = np.load("scripts/render/smpl_uv_map.npy")  # Shape (6890, 2)
    v_uv = torch.from_numpy(v_uv).to(device, dtype=torch.float32)   # Shape (1, 6890, 2)

    # Debug: Print shapes and a few samples to verify
    print("Faces shape:", faces.shape)  # Expected: (13776, 3)
    print("Vertices shape:", vertices.shape)  # Expected: (6890, 3)
    print("UV shape:", v_uv.shape)  # Expected: (6890, 2)
    print("Image shape:", img.shape)  # Expected: (H, W, 3)

    # Add batch dimensions as required for TexturesUV

    verts_uvs = v_uv.unsqueeze(0).repeat(256,1,1)  # Shape: (1, 6890, 2)
    faces_uvs = faces.unsqueeze(0).repeat(256,1,1)
    vertices = vertices.unsqueeze(0).repeat(256,1,1)
    faces = faces.unsqueeze(0).repeat(256,1,1)
    # Shape: (1, 13776, 3)
    img = img.unsqueeze(0).repeat(256,1,1,1)
    textures = TexturesUV(maps=img, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
    textures = textures.to(device)
    smpl_mesh = Meshes(
        verts=vertices,
        faces=faces,
        textures=textures
    )

    start_time=time.time()
    renderer = renderer.to(device)
    images = renderer(smpl_mesh)

    end_time = time.time()
    render_time = end_time-start_time

    print("render time is  ",render_time)
    plt.figure(dpi = 250)
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.grid("off");
    plt.axis("off")
    plt.show()




if __name__ == "__main__":
    main()





















