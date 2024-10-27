# offline training
python pde_2d_sac_train.py

# inference
python inference_2d.py \
--inference_method 'SAC' \
--batch_size 50 \
--sac_model_path '/cp/SAC_2024-05-22 10:02:38.283684_0_309'