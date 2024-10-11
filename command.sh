
#CUDA_VISIBLE_DEVICES=1 python phc/run_hydra.py env.task=HumanoidAnyskillZ \
#env=env_pulse_amp \
#exp_name=pulse_anyskill_kickleftlegforwardrightlegretreats \
#robot.real_weight_porpotion_boxes=False \
#learning=pulse_z_task \
#env.models=['output/HumanoidIm/pulse_vae_iclr/Humanoid.pth'] \
#env.motion_file=sample_data/amass_isaac_simple_run_upright_slim.pkl \
#env.num_envs=1024 \
#headless=True \
#no_virtual_display=False

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py  \
env.task=HumanoidSpeedZ \
env=env_pulse_amp \
exp_name=pulse_speed \
robot.real_weight_porpotion_boxes=False \
learning=pulse_z_task \
env.models=['output/HumanoidIm/pulse_vae_iclr/Humanoid.pth'] \
env.motion_file=sample_data/amass_isaac_simple_run_upright_slim.pkl \
env.num_envs=1024