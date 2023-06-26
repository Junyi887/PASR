export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --crop_size 512 \
                --batch_size 4 \
                --epochs 2 \
                --task_dt 1 \
                --loss_type "L1" \
                --data_path "../superbench/datasets/nskt16000_1024" \
                --lamb 1.0 & 
PID=$!
echo "PID for train.py: $PID" >> pid.log
wait $PID

