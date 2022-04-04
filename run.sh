export CUDA_VISIBLE_DEVICES=1
SOFT_TOKENS=100
TRAIN_PROMPT_STEPS=90
TRAIN_LABEL_STEPS=10
for LEARNING_RATE in 1e-2
do
    for T in 4
    do
        for NUM_LABELS in 1
        do
            python ./mask_tune.py \
                --seed 42 \
                --mask_tuning \
                --tune_label \
                --train_prompt_steps $TRAIN_PROMPT_STEPS \
                --train_label_steps $TRAIN_LABEL_STEPS \
                --base_path TextClassification \
                --dataset_name SST-2 \
                --model_name roberta \
                --model_path roberta-base \
                --use_cuda \
                --learning_rate $LEARNING_RATE \
                --train_batch_size 32 \
                --eval_batch_size 32 \
                --num_epochs 30 \
                --max_steps 6000 \
                --eval_steps 10 \
                --plot_steps 200 \
                --soft_tokens $SOFT_TOKENS \
                --num_labels $NUM_LABELS \
                --t $T \
                --log_name roberta-base-$LEARNING_RATE-$SOFT_TOKENS-soft-$NUM_LABELS-$T-tune_label-$TRAIN_PROMPT_STEPS-$TRAIN_LABEL_STEPS
        done
    done
done