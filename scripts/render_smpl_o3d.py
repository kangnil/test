import glob
import os
import sys
import pdb
import os.path as osp
from PIL import Image
sys.path.append(os.getcwd())

import open3d as o3d
import open3d.visualization.rendering as rendering
import imageio
from tqdm import tqdm
import joblib
import numpy as np
import torch

from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import random

from smpl_sim.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

paused, reset, recording, image_list, writer, control, curr_zoom = False, False, False, [], None, None, 0.01



def main():
    # Load SMPL data
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
        # vertices, joints = smpl_parser.get_joints_verts(pose=torch.from_numpy(pose_aa), th_betas=torch.from_numpy(beta[None,]))
    vertices = vertices.numpy() #99,6890,3
    faces = smpl_parser.faces #13776,3. min 0 max 6889

    # Initialize Open3D Visualizer with Key Callback
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=800, height=600)

    # Add ground plane
    box = o3d.geometry.TriangleMesh()
    ground_size, height = 5, 0.01
    box = box.create_box(width=ground_size, height=height, depth=ground_size)
    box.translate(np.array([-ground_size / 2, -height, -ground_size / 2]))
    box.compute_vertex_normals()

    box.rotate(sRot.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix())
    vis.add_geometry(box)

    path_to_image = "scripts/render/nongrey_male_0540.jpg"
    img = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)

    # create the uv coordinates
    v_uv = np.load("scripts/render/smpl_uv_map.npy") #6890*2
    v_uv[:, 1] = 1.0 - v_uv[:, 1]

    # assign the texture to the mesh



    # Add SMPL human mesh
    smpl_mesh = o3d.geometry.TriangleMesh()
    smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[0])
    smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
    smpl_mesh.compute_vertex_normals()
    smpl_mesh.textures = [o3d.geometry.Image(img)]
    # Flatten the UV coordinates for triangle faces
    triangle_uvs = []
    for face in faces:
        triangle_uvs.append(v_uv[face[0]])
        triangle_uvs.append(v_uv[face[1]])
        triangle_uvs.append(v_uv[face[2]])


    #uv = np.array([[0.2,0.2]] * (3 * len(smpl_mesh.triangles))) #41328,2

    smpl_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    smpl_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))


    vis.add_geometry(smpl_mesh)

    # Set Camera parameters
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 1])
    ctr.set_front([0, -3, 1])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.8)

    # Define callback functions to interactively change frame
    current_frame = [0]
    total_frames = vertices.shape[0]

    def update_mesh(vis):
        # Update mesh vertices for current frame
        smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[current_frame[0]])
        smpl_mesh.compute_vertex_normals()
        vis.update_geometry(smpl_mesh)
        vis.poll_events()
        vis.update_renderer()
        # Move to the next frame
        current_frame[0] = (current_frame[0] + 1) % total_frames

    # Register a callback to update the mesh on each key press
    vis.register_key_callback(ord("N"), update_mesh)

    # Start the visualizer
    vis.run()
    vis.destroy_window()




if __name__ == "__main__":
    main()