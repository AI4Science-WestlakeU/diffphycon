# Full observation partial control 
## online
python burgers_sac_train_quarter.py --reward_f 0 -online 
## offline
python burgers_sac_train_quarter.py --reward_f 0 
## pseudo-online
python burgers_sac_train_quarter.py --reward_f 0 --online --surrogate

# Paitial observation full control 
## online
burgers_sac_pob_fctr.py --reward_f 0 --online
## offline
burgers_sac_pob_fctr.py --reward_f 0
## pseudo-online
burgers_sac_pob_fctr.py --reward_f 0 --online --surrogate

# Paitial observation partial control 
## online
burgers_sac_pob_pctr.py --reward_f 0 --online
## offline
burgers_sac_pob_pctr.py --reward_f 0
## pseudo-online
burgers_sac_pob_pctr.py --reward_f 0 --online --surrogate
