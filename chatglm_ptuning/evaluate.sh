PRE_SEQ_LEN=64
CHECKPOINT=adgen-chatglm-6b-pt-64-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_predict \
    --validation_file eval.json \
    --test_file eval.json \
    --overwrite_cache \
    --prompt_column user1 \
    --response_column user2 \
    --model_name_or_path E:\\westlake\\chatglm-6b-int4 \
    --ptuning_checkpoint ./output/checkpoint-3000 \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
