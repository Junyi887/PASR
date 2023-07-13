export CUDA_VISIBLE_DEVICES=0
# correspond to real dt 
python eval.py --crop_size 128 \
                --model "PASR_MLP" \
                --batch_size 8 \
                --task_dt 0.2 \
                --ode_step 8 \
                --epoch 140 \
                --n_snapshot 10\
                --loss_type "L1" \
                --ode_method "RK4" \
                --data_path "../superbench/datasets/nskt16000_1024" \
                --lamb 0.3 & 
PID=$!
echo "PID for train.py: $PID" >> pid.log
wait $PID

