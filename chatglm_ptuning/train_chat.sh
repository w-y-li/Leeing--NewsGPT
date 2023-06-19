PRE_SEQ_LEN=64
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train \
    --train_file train.json \
    --validation_file eval.json \
    --prompt_column user1 \
    --response_column user2 \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path E:\\westlake\\chatglm-6b-int4 \
    --output_dir output2 \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
exec /bin/bash
