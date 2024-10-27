# Push data into memory
python pde_2d_sac_data_processing.py \
--dataset_path 'data/jellyfish' \
--image_size 64 


# Full observation\
## train
### online
python pde_2d_sac_train.py \
--dataset_path 'data/jellyfish' \
--image_size 64 \
--simulator_model_checkpoint "checkpoints/sim_new_epoch_9.pth" \
--boundary_updater_model_checkpoint "checkpoints/bd_epoch_3.pth" \
--force_model_checkpoint 'checkpoints/force_epoch_9.pth' \
--critic_lr 0.0003 \
--ent_lr 0.0003 \
--policy_lr 0.0003 \
--reward_alpha 0.001 \
--reg_lambda -1000 \
--online 

### offline
python pde_2d_sac_train.py \
--dataset_path 'data/jellyfish' \
--image_size 64 \
--simulator_model_checkpoint "checkpoints/sim_new_epoch_9.pth" \
--boundary_updater_model_checkpoint "checkpoints/bd_epoch_3.pth" \
--force_model_checkpoint 'checkpoints/force_epoch_9.pth' \
--critic_lr 0.0003 \
--ent_lr 0.0003 \
--policy_lr 0.0003 \
--reward_alpha 0.001 \
--reg_lambda -1000 

## inference (change sac_model_path and inference_result_path)
### online
python inference_2d.py \
--dataset_path 'data/jellyfish' \
--image_size 64 \
--sac_model_path 'model/2d/SAC_policy_track20_2024-01-15 14:32:48.301491_0_97' \
--simulator_model_checkpoint "checkpoints/sim_new_epoch_9.pth" \
--boundary_updater_model_checkpoint "checkpoints/bd_epoch_3.pth" \
--force_model_checkpoint 'checkpoints/force_epoch_9.pth' \
--inference_result_path 'results/2d_inf/' \
--inference_method 'SAC' 

### offline
python inference_2d.py \
--dataset_path 'data/jellyfish' \
--image_size 64 \
--sac_model_path 'model/2d/SAC_policy_track20_2024-01-14 14:18:31.214305_0_277' \
--simulator_model_checkpoint "checkpoints/sim_new_epoch_9.pth" \
--boundary_updater_model_checkpoint "checkpoints/bd_epoch_3.pth" \
--force_model_checkpoint 'checkpoints/force_epoch_9.pth' \
--inference_result_path 'results/2d_inf/' \
--inference_method 'SAC' 


# Partial observation
## train
### online
python pde_2d_sac_train_pob.py \
--dataset_path 'data/jellyfish' \
--image_size 64 \
--only_vis_pressure \
--simulator_model_checkpoint "checkpoints/sim_pob_epoch_9.pth" \
--boundary_updater_model_checkpoint "checkpoints/bd_epoch_3.pth" \
--force_model_checkpoint 'checkpoints/force_epoch_9.pth' \
--critic_lr 0.0003 \
--ent_lr 0.0003 \
--policy_lr 0.0003 \
--reward_alpha 0.001 \
--reg_lambda -1000 \
--online 

### offline
python pde_2d_sac_train_pob.py \
--dataset_path 'data/jellyfish' \
--image_size 64 \
--only_vis_pressure \
--simulator_model_checkpoint "checkpoints/sim_pob_epoch_9.pth" \
--boundary_updater_model_checkpoint "checkpoints/bd_epoch_3.pth" \
--force_model_checkpoint 'checkpoints/force_epoch_9.pth' \
--critic_lr 0.0003 \
--ent_lr 0.0003 \
--policy_lr 0.0003 \
--reward_alpha 0.001 \
--reg_lambda -1000

## inference (change sac_model_path and inference_result_path)
### online
python inference_2d.py \
--dataset_path 'data/jellyfish' \
--image_size 64 \
--sac_model_path 'model/2d/SAC_policy_track20_pob_2024-01-14 15:12:55.301397_0_261' \
--simulator_model_checkpoint "checkpoints/sim_pob_epoch_9.pth" \
--boundary_updater_model_checkpoint "checkpoints/bd_epoch_3.pth" \
--force_model_checkpoint 'checkpoints/force_epoch_9.pth' \
--inference_result_path 'results/2d_inf/' \
--inference_method 'SAC' \
--only_vis_pressure

### offline
python inference_2d.py \
--dataset_path 'data/jellyfish' \
--image_size 64 \
--sac_model_path 'model/2d/SAC_policy_track20_pob_2024-01-14 15:16:24.012008_0_313' \
--simulator_model_checkpoint "checkpoints/sim_pob_epoch_9.pth" \
--boundary_updater_model_checkpoint "checkpoints/bd_epoch_3.pth" \
--force_model_checkpoint 'checkpoints/force_epoch_9.pth' \
--inference_result_path 'results/2d_inf/' \
--inference_method 'SAC' \
--only_vis_pressure