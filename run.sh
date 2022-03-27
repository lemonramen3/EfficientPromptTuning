export CUDA_VISIBLE_DEVICES=7

for LEARNING_RATE in 1e-2
do
    for T in 5
    do
        for NUM_LABELS in 1
        do
            python ./mask_tune.py \
                --seed 42 \
                --mask_tuning \
                --base_path TextClassification \
                --dataset_name SST-2 \
                --model_name roberta \
                --model_path roberta-base \
                --use_cuda \
                --learning_rate $LEARNING_RATE \
                --train_batch_size 32 \
                --eval_batch_size 64 \
                --num_epochs 3 \
                --max_steps 2000 \
                --eval_steps 10 \
                --plot_steps 100 \
                --num_labels $NUM_LABELS \
                --t $T \
                --log_name roberta-base-test
        done
    done
done