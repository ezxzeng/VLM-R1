from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from open_r1.qwen_vl_utils import process_vision_info
from open_r1.counting_rewards import (
    counting_format_reward,
    points_reward,
    point_accuracy_reward,
    count_consistency_reward,
    counting_accuracy_reward,
    segmentation_reward
)
import torch
import json
import re
import os
import random
from tqdm import tqdm
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model for object counting")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint", default="/home/ezzeng/stat946_final_project/VLM-R1/src/open-r1-multimodal/output/Qwen2.5-VL-3B-GRPO-TALLY-lora-ratio-cntFormR/checkpoint-350")
    parser.add_argument("--data_path", type=str, help="Path to the test data (JSONL)", default="/home/ezzeng/stat946/stat946_final_proj/data/coco/val_qas_subset.jsonl")
    parser.add_argument("--image_root", type=str, help="Path to image directory", default="/home/ezzeng/stat946/stat946_final_proj/data")
    parser.add_argument("--output_path", type=str, help="Path to save results", default="/home/ezzeng/stat946/stat946_final_proj/logs/")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--segmentation_folder", type=str, help="Path to segmentation folder", default="/home/ezzeng/stat946/stat946_final_proj/data/coco/segmentation_masks")
    return parser.parse_args()

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return local_rank, world_size, rank

def load_jsonl(file_path, num_samples=None):
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    # Optionally limit the number of samples
    if num_samples and len(data) > num_samples:
        random.seed(42)
        random.shuffle(data)
        data = data[:num_samples]
    
    return data

def main(args, local_rank=0, world_size=1, rank=0):
    device = f"cuda:{local_rank}"
    
    output_dir = os.path.join(args.output_path, ("_").join(args.model_path.split("/")[-2:]))
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a streaming log file for immediate results
    streaming_log_path = os.path.join(output_dir, f"streaming_results_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    
    if rank == 0:
        print(f"Process {rank} using {device}")
        print(f"Loading model from {args.model_path}")
        print(f"Streaming results will be written to {streaming_log_path}")
    
    # Load the base model first
    print(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank},
    )
    
    # Check if this is a LoRA model by looking for adapter_config.json
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        if rank == 0:
            print("Loading LoRA adapter...")
        # For LoRA models, load the adapter
        model = PeftModel.from_pretrained(model, args.model_path)
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    if rank == 0:
        print(f"Loading test data from {args.data_path}")
    
    data = load_jsonl(args.data_path, args.num_samples)
    
    # Split data for distributed evaluation
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]
    
    if rank == 0:
        print(f"Processing {len(data)} examples across {world_size} processes")
    
    # Define the question template with think/answer instructions
    QUESTION_TEMPLATE = "First, count each instance you spot and their <x, y> coordinates as part of the thinking process in <think> </think> tags. Then output the final answer as a single number in <answer> </answer> tags. Below are some examples: \n <think>I spot a person at <401, 105>.\nI spot a person at <48, 19>.\nTherefore, I see 2 people.</think><answer>2</answer> \n <think>I spot an apple at <352, 258>.\nI spot an apple at <45, 351>.\nTherefore, I see 2 apples.</think><answer>2</answer> \n <think>I spot a pizza at <220, 421>.\nTherefore, I see 1 pizza.</think>\n<answer>1</answer> \n <think>I spot a cow at <118, 112>.\nI spot a cow at <143, 207>.\nI spot a cow at <356, 41>.\nTherefore, I see 3 cows.</think>\n<answer>3</answer>. \n {Question}."
    
    # Prepare messages for the model
    messages = []
    original_examples = []  # Store original examples for result logging
    
    for x in rank_data:
        original_examples.append(x)
        
        # Handle case where image is a string or a list
        if isinstance(x["image"], list):
            image_paths = [os.path.join(args.image_root, img) for img in x["image"]]
            image_content = [{"type": "image", "image": f"file://{path}"} for path in image_paths]
        else:
            image_path = os.path.join(args.image_root, x["image"])
            image_content = [{"type": "image", "image": f"file://{image_path}"}]
        
        # Extract question from conversations
        question = x["conversations"][0]["value"].replace("<image>", "")
        
        # Apply the question template
        formatted_question = QUESTION_TEMPLATE.format(Question=question)
        
        message = [{
            "role": "user",
            "content": [
                *image_content,
                {
                    "type": "text",
                    "text": formatted_question
                }
            ]
        }]
        messages.append(message)
    
    # To store interim correct counts for each rank
    correct_count = 0
    total_processed = 0
    
    # Process data in batches
    rank_outputs = []

    rank_counting_format = []
    rank_points = []
    rank_point_accuracy = []
    rank_count_consistency = []
    rank_counting_accuracy = []
    rank_segmentation = []
    for i in tqdm(range(0, len(messages), args.batch_size), disable=rank != 0):
        batch_messages = messages[i:i + args.batch_size]
        batch_examples = original_examples[i:i + args.batch_size]
        
        # Prepare inputs for the model
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        # Use a safer approach for processor call
        processor_inputs = {
            "text": text,
            "images": image_inputs,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        
        # Only add videos if they exist
        if video_inputs is not None:
            processor_inputs["videos"] = video_inputs
            
        inputs = processor(**processor_inputs)
        inputs = inputs.to(device)
        
        # Generate outputs
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        rank_outputs.extend(batch_output_text)

        completions = [[{"content": output}] for output in batch_output_text]
        solution = [example["conversations"][1]["value"] for example in batch_examples]
        segmentation_paths = [os.path.join(args.segmentation_folder, example["segmentation_path"]) for example in batch_examples]

        _counting_format = counting_format_reward(completions)
        _points = points_reward(completions, solution)
        _point_accuracy = point_accuracy_reward(completions, solution)
        _count_consistency = count_consistency_reward(completions, solution)
        _counting_accuracy = counting_accuracy_reward(completions, solution)
        _segmentation = segmentation_reward(completions, solution, segmentation_path=segmentation_paths)

        rank_counting_format.extend(_counting_format)
        rank_points.extend(_points)
        rank_point_accuracy.extend(_point_accuracy)
        rank_count_consistency.extend(_count_consistency)
        rank_counting_accuracy.extend(_counting_accuracy)
        rank_segmentation.extend(_segmentation)
        
        # write results to jsonl file
        for i in range(len(batch_examples)):
            question_id = batch_examples[i]["id"]
            with open(os.path.join(output_dir, f"{question_id}.json"), "w") as f:
                json.dump({
                    "completion": completions[i],
                    "solution": solution[i],
                    "segmentation_path": segmentation_paths[i],
                    "counting_format": _counting_format[i],
                    "points": _points[i],
                    "point_accuracy": _point_accuracy[i],
                    "count_consistency": _count_consistency[i],
                    "counting_accuracy": _counting_accuracy[i],
                    "segmentation_reward": _segmentation[i]
                }, f, indent=2)
    
    if rank == 0:
        print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")
    

if __name__ == "__main__":
    args = parse_args()

    model_paths = [
        "/pub5/ezzeng/stat946_final_project/output/Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR-pointsR-segR-pointAccR-cntConsR-cntAccR/checkpoint-579",
        "/pub5/ezzeng/stat946_final_project/output/Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR-pointsR-pointAccR-cntConsR-cntAccR/checkpoint-579",
        "/pub5/ezzeng/stat946_final_project/output/Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR-pointsR-segR-cntConsR-cntAccR/checkpoint-579",
        "/pub5/ezzeng/stat946_final_project/output/Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR/checkpoint-579",
        "/pub5/ezzeng/stat946_final_project/output/Qwen2.5-VL-3B-GRPO-TALLY-lora-cntConsR/checkpoint-579",
        "/pub5/ezzeng/stat946_final_project/output/Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR-segR/checkpoint-579",
        "/pub5/ezzeng/stat946_final_project/output/Qwen2.5-VL-3B-GRPO-TALLY-lora-cntAccR/checkpoint-579",
        "/pub5/ezzeng/stat946_final_project/output/Qwen2.5-VL-3B-GRPO-TALLY-lora-segR/checkpoint-579"
    ]  

    for model_path in model_paths:
        args.model_path = model_path
        main(args, local_rank=1)