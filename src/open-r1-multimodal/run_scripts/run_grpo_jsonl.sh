cd src/open-r1-multimodal


# in host 'narval'
if [[ $(hostname) == *"narval"* ]]; then
  # Loading the required modules
  module --force purge && module load StdEnv/2023 gcc opencv/4.10.0 python/3.10.13 rust/1.85.0 arrow/18.1.0 cuda/12.6 && source ~/.venvs/vlm-r1/bin/activate

  # Setting offline mode
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export WANDB_MODE=offline
fi


export DEBUG_MODE="true"
export RUN_NAME="Qwen2.5-VL-3B-GRPO-TALLY-lora-count"
export LOG_PATH="output/$RUN_NAME/debug_log_$RUN_NAME.txt"
if [[ $(hostname) != *"narval"* ]]; then
  export TORCH_DYNAMO=0
  export CUDA_VISIBLE_DEVICES="1,2"
fi

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
  src/open_r1/grpo_jsonl.py \
    --question_task_template count \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed local_scripts/zero2.json \
    --dataset_name tally_qa \
    --data_file_paths /home/ezzeng/stat946/stat946_final_proj/data/train_qas.jsonl \
    --image_folders /home/ezzeng/stat946/stat946_final_proj/data \
    --max_prompt_length 1024 \
    --num_generations 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true