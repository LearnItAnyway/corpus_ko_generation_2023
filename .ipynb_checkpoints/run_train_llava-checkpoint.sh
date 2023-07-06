deepspeed train_llava.py \
    --model_name_or_path ./KoLLaVA-KoVicuna-7b \
    --data_path ./2023_nlg_combined.json \
    --output_dir ./output \
    --num_train_epochs 1 \
    --model_max_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_steps 1 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --fp16 True --half_precision_backend 'cuda_amp' \
    --deepspeed ./ds_config.json  \
    --gradient_accumulation_steps 128  \

python eval_2.py
python eval_5.py
