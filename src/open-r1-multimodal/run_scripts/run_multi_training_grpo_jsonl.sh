cd src/open-r1-multimodal
export DEBUG_MODE="true"
export TORCH_DYNAMO=0
export CUDA_VISIBLE_DEVICES="1,2"



export RUN_NAME="Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR-pointsR-segR-pointAccR-cntConsR-cntAccR"
export LOG_PATH="output/$RUN_NAME/debug_log.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
  src/open_r1/grpo_jsonl.py \
    --question_task_template count_many_examples_question_last \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed local_scripts/zero2.json \
    --dataset_name tally_qa \
    --data_file_paths /home/ezzeng/stat946/stat946_final_proj/data/coco/train_qas_subset.jsonl \
    --image_folders /home/ezzeng/stat946/stat946_final_proj/data \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --segmentation_folder /home/ezzeng/stat946/stat946_final_proj/data/coco/segmentation_masks \
    --reward_funcs format counting_format_reward points count_consistency counting_accuracy segmentation_reward point_accuracy


export RUN_NAME="Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR-segR"
export LOG_PATH="output/$RUN_NAME/debug_log.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
  src/open_r1/grpo_jsonl.py \
    --question_task_template count_many_examples_question_last \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed local_scripts/zero2.json \
    --dataset_name tally_qa \
    --data_file_paths /home/ezzeng/stat946/stat946_final_proj/data/coco/train_qas_subset.jsonl \
    --image_folders /home/ezzeng/stat946/stat946_final_proj/data \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --segmentation_folder /home/ezzeng/stat946/stat946_final_proj/data/coco/segmentation_masks \
    --reward_funcs counting_format_reward segmentation_reward

export RUN_NAME="Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR"
export LOG_PATH="output/$RUN_NAME/debug_log.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
  src/open_r1/grpo_jsonl.py \
    --question_task_template count_many_examples_question_last \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed local_scripts/zero2.json \
    --dataset_name tally_qa \
    --data_file_paths /home/ezzeng/stat946/stat946_final_proj/data/coco/train_qas_subset.jsonl \
    --image_folders /home/ezzeng/stat946/stat946_final_proj/data \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --segmentation_folder /home/ezzeng/stat946/stat946_final_proj/data/coco/segmentation_masks \
    --reward_funcs counting_format_reward


export RUN_NAME="Qwen2.5-VL-3B-GRPO-TALLY-lora-pointsR"
export LOG_PATH="output/$RUN_NAME/debug_log.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
  src/open_r1/grpo_jsonl.py \
    --question_task_template count_many_examples_question_last \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed local_scripts/zero2.json \
    --dataset_name tally_qa \
    --data_file_paths /home/ezzeng/stat946/stat946_final_proj/data/coco/train_qas_subset.jsonl \
    --image_folders /home/ezzeng/stat946/stat946_final_proj/data \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --segmentation_folder /home/ezzeng/stat946/stat946_final_proj/data/coco/segmentation_masks \
    --reward_funcs points


export RUN_NAME="Qwen2.5-VL-3B-GRPO-TALLY-lora-segR"
export LOG_PATH="output/$RUN_NAME/debug_log.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
  src/open_r1/grpo_jsonl.py \
    --question_task_template count_many_examples_question_last \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed local_scripts/zero2.json \
    --dataset_name tally_qa \
    --data_file_paths /home/ezzeng/stat946/stat946_final_proj/data/coco/train_qas_subset.jsonl \
    --image_folders /home/ezzeng/stat946/stat946_final_proj/data \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --segmentation_folder /home/ezzeng/stat946/stat946_final_proj/data/coco/segmentation_masks \
    --reward_funcs segmentation_reward


export RUN_NAME="Qwen2.5-VL-3B-GRPO-TALLY-lora-pointAccR"
export LOG_PATH="output/$RUN_NAME/debug_log.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
  src/open_r1/grpo_jsonl.py \
    --question_task_template count_many_examples_question_last \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed local_scripts/zero2.json \
    --dataset_name tally_qa \
    --data_file_paths /home/ezzeng/stat946/stat946_final_proj/data/coco/train_qas_subset.jsonl \
    --image_folders /home/ezzeng/stat946/stat946_final_proj/data \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --segmentation_folder /home/ezzeng/stat946/stat946_final_proj/data/coco/segmentation_masks \
    --reward_funcs point_accuracy


export RUN_NAME="Qwen2.5-VL-3B-GRPO-TALLY-lora-cntConsR"
export LOG_PATH="output/$RUN_NAME/debug_log.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
  src/open_r1/grpo_jsonl.py \
    --question_task_template count_many_examples_question_last \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed local_scripts/zero2.json \
    --dataset_name tally_qa \
    --data_file_paths /home/ezzeng/stat946/stat946_final_proj/data/coco/train_qas_subset.jsonl \
    --image_folders /home/ezzeng/stat946/stat946_final_proj/data \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --segmentation_folder /home/ezzeng/stat946/stat946_final_proj/data/coco/segmentation_masks \
    --reward_funcs count_consistency


export RUN_NAME="Qwen2.5-VL-3B-GRPO-TALLY-lora-cntAccR"
export LOG_PATH="output/$RUN_NAME/debug_log.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
  src/open_r1/grpo_jsonl.py \
    --question_task_template count_many_examples_question_last \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed local_scripts/zero2.json \
    --dataset_name tally_qa \
    --data_file_paths /home/ezzeng/stat946/stat946_final_proj/data/coco/train_qas_subset.jsonl \
    --image_folders /home/ezzeng/stat946/stat946_final_proj/data \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --segmentation_folder /home/ezzeng/stat946/stat946_final_proj/data/coco/segmentation_masks \
    --reward_funcs counting_accuracy