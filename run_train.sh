export CUDA_VISIBLE_DEVICES=0,1,2
python train.py --crop_size 1024 \
                --batch_size 1 \
                --epochs 2 \
                --task_dt 1 \
                --loss_type "L1" \
                --data_path "../superbench/dataset/nskt16000_1024" \
                --lamb 1.0 & 
PID=$!
echo "PID for train.py: $PID" >> pid.log
wait $PID

