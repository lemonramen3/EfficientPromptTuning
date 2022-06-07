# run.sh
export CUDA_VISIBLE_DEVICES=0
MODEL_NAME=roberta
MODEL_PATH=roberta-base
for LEARNING_RATE in 1e-3
do
    for T in 0.9
    do
        for NUM_LABELS in 5
        do
            for SOFT_TOKENS in 20
            do
                for TRAIN_PROMPT_STEPS in 200
                do
                    for TRAIN_LABEL_STEPS in 10
                    do
                        for RECLUSTER_STEPS in 3500
                        do
                            python ./mask_tune.py \
                                --seed 42 \
                                --base_path TextClassification \
                                --dataset_name SST-2 \
                                --mask_tuning \
                                --tune_label \
                                --train_label_steps $TRAIN_LABEL_STEPS \
                                --train_prompt_steps $TRAIN_PROMPT_STEPS \
                                --model_name $MODEL_NAME \
                                --model_path $MODEL_PATH \
                                --use_cuda \
                                --learning_rate $LEARNING_RATE \
                                --train_batch_size 32 \
                                --eval_batch_size 32 \
                                --num_epochs 30 \
                                --max_steps 5000 \
                                --eval_steps 10 \
                                --plot_steps 5000 \
                                --soft_tokens $SOFT_TOKENS \
                                --num_labels $NUM_LABELS \
                                --t $T \
                                --log_name num_labels=$NUM_LABELS-recluster_steps=$RECLUSTER_STEPS
                        done
                    done
                done
            done
        done
    done
done