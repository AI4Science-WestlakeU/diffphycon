source /opt/conda/bin/activate
mkdir -p trained_models/{burgers,burgers_w}/

# train model of p(u, w)
python train/train_1d_burgers.py \
    --is_condition_u0 True \
    --is_condition_uT True \
    --exp_id POPC \
    --dim 64 \
    --dataset free_u_f_1e5_front_rear_quarter \
    --partially_observed front_rear_quarter \
    --train_on_partially_observed front_rear_quarter \
    --dim_muls 1 2 4 8 \
    --train_num_steps 200000 \
    --checkpoint_interval 1000

# train model of p(w)
python train/train_1d_burgers.py \
    --is_condition_u0 True \
    --is_condition_uT True \
    --exp_id POPC_w \
    --dim 64 \
    --dataset free_u_f_1e5_front_rear_quarter \
    --partially_observed front_rear_quarter \
    --train_on_partially_observed front_rear_quarter \
    --dim_muls 1 2 4 8 \
    --train_num_steps 200000 \
    --checkpoint_interval 1000 \
    --is_model_w True \
    --expand_condition False