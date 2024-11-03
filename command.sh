

# python phc/run_hydra.py  env.task=HumanoidAnyskillZ env=env_pulse_amp exp_name=pulse_motionfile_standing_text_a_person_is_running_xl robot.real_weight_porpotion_boxes=False \
#  learning=pulse_z_task  env.models='['output/HumanoidIm/pulse_vae_iclr/Humanoid.pth']' env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
#   env.num_envs=256 im_eval=True env.text_command="a person is running" env.llm_model='clip-flant5-xl' env.llm_model_device='cuda:1' env.llm_model_batchsize=128


# python phc/run_hydra.py  env.task=HumanoidAnyskillZ env=env_pulse_amp exp_name=pulse_motionfile_standing_text_a_person_is_kicking_xl robot.real_weight_porpotion_boxes=False \
#  learning=pulse_z_task  env.models='['output/HumanoidIm/pulse_vae_iclr/Humanoid.pth']' env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
#   env.num_envs=256 im_eval=True env.text_command="a person is kicking" env.llm_model='clip-flant5-xl' env.llm_model_device='cuda:2' env.llm_model_batchsize=128

python phc/run_hydra.py  env.task=HumanoidAnyskillZ env=env_pulse_amp exp_name=pulse_motionfile_standing_text_a_person_is_clapping_xxl robot.real_weight_porpotion_boxes=False \
 learning=pulse_z_task  env.models='['output/HumanoidIm/pulse_vae_iclr/Humanoid.pth']' env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=256 im_eval=True env.text_command="a person is clapping" env.llm_model='clip-flant5-xxl' env.llm_model_device='cuda:2' env.llm_model_batchsize=128  

export OMP_NUM_THREADS=1 
python phc/run_hydra.py  env.task=HumanoidAnyskillZ env=env_pulse_amp exp_name=pulse_motionfile_standing_text_a_person_is_bowing_xxl robot.real_weight_porpotion_boxes=False \
 learning=pulse_z_task  env.models='['output/HumanoidIm/pulse_vae_iclr/Humanoid.pth']' env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=256 im_eval=True env.text_command="a person is bowing" env.llm_model='clip-flant5-xxl' env.llm_model_device='cuda:1' env.llm_model_batchsize=128



