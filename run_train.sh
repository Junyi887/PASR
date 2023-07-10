export CUDA_VISIBLE_DEVICES=0,1
# correspond to real dt 
python eval.py --crop_size 128 \
                --batch_size 32 \
                --task_dt 0.2 \
                --loss_type "L1" \
                --data_path "../superbench/datasets/nskt16000_1024" \
                --lamb 0.3 & 
PID=$!
echo "PID for train.py: $PID" >> pid.log
wait $PID

