source /opt/conda/bin/activate

# inference of DiffPhyCon
python inference/inference_1d_burgers.py \
    --dataset free_u_f_1e5_front_rear_quarter \
    --partial_control front_rear_quarter \
    --partially_observed None \
    --train_on_partially_observed None \
    --set_unobserved_to_zero_during_sampling False \
    --is_condition_u0 True \
    --is_condition_uT True \
    --J_scheduler cosine \
    --dim 64 \
    --dim_muls 1 2 4 \
    --exp_id FOPC \
    --checkpoint_interval 1000\
    --checkpoint 170 \
    --dim__model_w 32 \
    --dim_muls__model_w 1 2 4 8 \
    --exp_id__model_w FOPC_w \
    --checkpoint_interval__model_w 1000\
    --checkpoint__model_w 90 \
    --save_file burgers_results/full_obs_partial_ctr/result.yaml \
    --is_model_w False \
    --eval_two_models True \
    --expand_condition False \
    --prior_beta 1.5 \
    --normalize_beta False \
    --w_scheduler sigmoid_flip \
    --wfs 0

# eval DiffPhyCon-lite
python inference/inference_1d_burgers.py \
    --exp_id FOPC \
    --dataset free_u_f_1e5_front_rear_quarter \
    --is_condition_u0 True \
    --is_condition_uT True \
    --J_scheduler cosine \
    --dim 64 \
    --dim_muls 1 2 4 \
    --partial_control front_rear_quarter \
    --partially_observed None \
    --train_on_partially_observed None \
    --set_unobserved_to_zero_during_sampling False \
    --checkpoint_interval 1000\
    --checkpoint 170 \
    --save_file burgers_results/full_obs_partial_ctr/result_lite.yaml
