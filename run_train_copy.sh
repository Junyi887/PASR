export CUDA_VISIBLE_DEVICES=2,3
python train.py --crop_size 256 \
                --batch_size 8 \
                --epochs 200 \
                --task_dt 1 \
                --loss_type "L1" \
                --data_path "../superbench/datasets/nskt16000_1024" \
                --lamb 1 & 
PID=$!
echo "PID for train.py: $PID" >> pid.log
wait $PID

