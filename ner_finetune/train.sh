torchrun --nproc_per_node=2 --master_port=12345 train.py \
    --model_name_or_path '/home/quangng/LLM/KnowlabLLM-NER/models/phi3-4k-wikiterms-pretrain' \
    --data_path ./instructions_bioner.json \
    --bf16 False \
    --fp16 True \
    --max_steps 100 \
    --output_dir 'models/Phi-3-mini-4k-wikiterms-instruct-finetuned' \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --report_to "tensorboard" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --optim "adamw_torch" \
    --max_grad_norm 1.0 \
    --tf32 True

DATA_TRAIN="scilay"
torchrun --nproc_per_node=2 --master_port=12345 train.py \
    --model_name_or_path "/home/quangng/LLM/KnowlabLLM-NER/models/phi3-4k-${DATA_TRAIN}-pretrain" \
    --data_path ./instructions_bioner.json \
    --bf16 False \
    --fp16 True \
    --max_steps 100 \
    --output_dir "models/phi-3-mini-4k-${DATA_TRAIN}-instruct-finetuned" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --report_to "tensorboard" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --optim "adamw_torch" \
    --max_grad_norm 1.0 \
    --tf32 True


MODEL="KnowMedPhi3-medium-it"
torchrun --nproc_per_node=2 --master_port=12345 train.py \
    --model_name_or_path "/home/quangng/LLM/KnowlabLLM-NER/models/${MODEL}" \
    --data_path ./data/instructions_bioner.json \
    --bf16 False \
    --fp16 True \
    --max_steps 100 \
    --output_dir "models/${MODEL}-finetuned" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --report_to "tensorboard" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --optim "adamw_torch" \
    --max_grad_norm 1.0 \
    --tf32 True
