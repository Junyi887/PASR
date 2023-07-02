export CUDA_VISIBLE_DEVICES=2,3
# change of lambda
python train.py --crop_size 128 \
                --batch_size 32 \
                --epochs 150 \
                --loss_type "L1" \
                --data_path "../superbench/datasets/nskt16000_1024" \
                --lamb 1 & 
PID=$!
echo "PID for train.py: $PID" >> pid.log
wait $PID

