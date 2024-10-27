for var in $(seq $1 $2)
do
    python 3_2_burgers_control.py 'result_100' --data_num $var --lamb1 10 --lamb2 0.3 --ftol 4e-6 --lr 5e-1
done