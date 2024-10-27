# pseudo-online training
python pde_2d_sac_train.py --online

# inference
python inference_2d.py \
--inference_method 'SAC' \
--batch_size 50 \
--sac_model_path '/cp/SAC_2024-05-22 10:03:29.734402_0_89'