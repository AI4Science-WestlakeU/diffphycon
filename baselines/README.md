# Baselines

## 1D Burgers' Equation:
### Surrogate model
```code
python /model/pde_1d_surrogate_model/burgers_operator.py --date_time "2024-01-09_1d_surrogate_partial_ob_partial_ctr_autoregress-3" --epochs 500 --gpu 0 --train_batch_size 151 --autoregress_steps 3 --dataset_path "/dataset_control_burgers/free_u_f_1e5_front_rear_quarter" --is_partially_observable 1 --is_partially_controllable 1 --lr 0.001
```
### PID
```code
python pde_1d_control_PID.py --date_time "2024-01-09_PID_solver_partial_clamp-" --is_partially_controllable 1 --is_partially_observable 0 --dataset_path "/dataset_control_burgers/free_u_f_1e5_front_rear_quarter" --max_training_iters 10000 --save_iters 1000 --gpuid 7 --simulation_method "surrogate_model" --pde_1d_surrogate_model_checkpoint "/pde_gen_control/checkpoints/pde_1d/full_ob_partial_ctr_1-step" --f_max 100
```
### SAC
```code
bash /sac_burgers/sac_burgers.sh
```

### SL
```code
bash /sl_burgers/sl_burgers.sh
```

### BC
```code
Train: bash baselines/BC_burgers/train.sh

Inference: bash baselines/BC_burgers/inf.sh
```

### BPPO
```code
Train: bash baselines/BPPO_burgers/train.sh

Inference: bash baselines/BPPO_burgers/inf.sh
```

## 2D Jellyfish:
### MPC
```code
python inference_2d.py --inference_result_path "/results_MPC_0119_full_ob/MPC_0119_iter-50_lamb-500_endcond_coef-500_coef_clip-1000/" \
--dataset_path  '/datasets/jellyfish/' \
--image_size 64 \
--force_model_checkpoint '/checkpoints/force_epoch_9.pth' \
--simulator_model_checkpoint "/checkpoints/epoch_9_no_mask.pth" \
--boundary_updater_model_checkpoint "/checkpoints/bd_epoch_3.pth" \
--num_iters 50 \
--coef_grad 0.0001 \
--coef_clip 1000 \
--coef_endcondition 500 \
--lamda 500 \
--batch_size 50 \
--num_batches 1 \
```
### SAC
```code
bash /sac_jellyfish/sac_jellyfish.sh
```
### SL
```code
XXX
```

### BC
```code
Train: bash baselines/BC_jellyfish/train.sh

Inference: bash baselines/BC_jellyfish/inf.sh
```

### BPPO
```code
Train: bash baselines/BPPO_jellyfish/train.sh

Inference: bash baselines/BPPO_jellyfish/inf.sh
```

## 2D Smoke:
### SAC
offline:
```code
bash /sac_smoke/sac_smoke_offline.sh
```
online:
```code
bash /sac_smoke/sac_smoke_pseudo_online.sh
```

### BC
```code
Train: bash baselines/BC_smoke/train.sh

Inference: bash baselines/BC_smoke/inf.sh
```

### BPPO
```code
Train: bash baselines/BPPO_smoke/train.sh

Inference: bash baselines/BPPO_smoke/inf.sh
```

