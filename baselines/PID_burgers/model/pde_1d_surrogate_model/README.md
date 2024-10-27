# Force control of Burgers’ equation

Every detail parameter is explained in each script.

## Data generation
```
python 1_data_gen.py
```

## Phase 1: Solution operator learning
```
python 2_burgers_operator.py
```
You can use the dataset, `data\burgers_dataset`.

## Phase 2: Searching for optimal control

Samples generation for control problem(u_0, u_d).
```
python 3_1_burgers_control_generate_samples.py
```
Find optimal control(`3_2_burgers_control.py`)
=> The result of our method and the numerical method (adjoint method) for 100 samples can be reproduced as follows:
```
bash burgers_control.sh 0 99
```
You can use the attached parameter, `logs/burgers_operator_model`.
Also, You can use the samples, `data/burgers_control_samples`

## Visual results
All figures in the paper can be reproduced in `4_vis_results.ipynb`.
