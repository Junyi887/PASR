export CUDA_VISIBLE_DEVICES=0
PID=114579
wait $PID
python train_rbc.py --crop_size 512 \
                --data "rbc_diff_IC" \
                --model "PASR_MLP" \
                --batch_size 2 \
                --task_dt 0.1 \
                --ode_step 8 \
                --epoch 150 \
                --n_snapshot 10\
                --loss_type "L1" \
                --ode_method "RK4" \
                --data_path "../datasets/rbc_diff_IC"\
                --lamb 1 & 
PID=$!
echo "PID for train.py: $PID" >> pid.log
wait $PID
