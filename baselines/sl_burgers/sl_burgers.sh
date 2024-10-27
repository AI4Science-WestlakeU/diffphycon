# Full observation, partial control
python sl_burgers_control.py --lamb1 500 --lamb2 0.03 --ftol 4e-7 --gtol 4e-7 --lr 1e-1 --reward_f 0
# partial observation, full control
python sl_burgers_pob_fctr.py --lamb1 50000 --lamb2 3 --ftol 4e-7 --gtol 4e-7 --lr 1e-1 --reward_f 0 
# partial observation, partial control
python sl_burgers_pob_pctr.py --lamb1 50000 --lamb2 3 --ftol 4e-7 --gtol 4e-7 --lr 1e-1 --reward_f 0 

