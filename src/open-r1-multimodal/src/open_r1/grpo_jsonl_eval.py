#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datasets import Dataset
from peft import PeftModel

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    HfArgumentParser,
    AutoModelForCausalLM
)

# Import specific model classes for VL models
try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    print("Qwen2VLForConditionalGeneration not available in your transformers version")

try:
    from transformers import Qwen2ForCausalLM
except ImportError:
    print("Qwen2ForCausalLM not available in your transformers version")

from open_r1.vlm_modules import *
from open_r1.counting_rewards import *

# Set up basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvalArguments:
    """Arguments for evaluation."""
    checkpoint_path: str = field(
        metadata={"help": "Path to the checkpoint directory"}
    )
    data_file_path: str = field(
        default="/home/ezzeng/stat946/stat946_final_proj/data/val_qas.jsonl",
        metadata={"help": "Path to validation data file"}
    )
    image_folder: str = field(
        default="/home/ezzeng/stat946/stat946_final_proj/data",
        metadata={"help": "Path to image folder"}
    )
    base_model_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Path to base model"}
    )
    output_file: str = field(
        default=None,
        metadata={"help": "Path to output file for results"}
    )
    question_task_template: str = field(
        default="count_many_examples_question_last",
        metadata={"help": "Question task template"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for evaluation"}
    )
    metrics: str = field(
        default="accuracy,format",
        metadata={"help": "Comma-separated list of metrics to evaluate"}
    )
    segmentation_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to segmentation folder"}
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate"}
    )
    top_p: float = field(
        default=0.7,
        metadata={"help": "Top-p sampling parameter"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature for sampling"}
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"}
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use a fast tokenizer"}
    )

def get_vlm_module(model_name_or_path):
    """Get the appropriate VLM module based on the model name"""
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def get_model_class(model_name_or_path):
    """Get the appropriate model class based on the model name"""
    if "qwen" in model_name_or_path.lower():
        if "vl" in model_name_or_path.lower():
            try:
                return Qwen2VLForConditionalGeneration
            except NameError:
                # Fall back to auto model if specific class isn't available
                logger.warning("Qwen2VLForConditionalGeneration not available, falling back to AutoModelForCausalLM")
                return AutoModelForCausalLM
        else:
            try:
                return Qwen2ForCausalLM
            except NameError:
                logger.warning("Qwen2ForCausalLM not available, falling back to AutoModelForCausalLM")
                return AutoModelForCausalLM
    else:
        return AutoModelForCausalLM

def load_model_and_processor(args):
    """Load model, tokenizer and processor"""
    logger.info(f"Loading model from {args.checkpoint_path}")
    
    # Load tokenizer and processor from base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_path,
            use_fast=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code
        )
        logger.info(f"Successfully loaded tokenizer from {args.base_model_path}")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

    processor = None
    if "qwen" in args.base_model_path.lower() and "vl" in args.base_model_path.lower():
        try:
            processor = AutoProcessor.from_pretrained(
                args.base_model_path,
                trust_remote_code=args.trust_remote_code
            )
            logger.info(f"Successfully loaded processor from {args.base_model_path}")
        except Exception as e:
            logger.error(f"Error loading processor: {e}")
            processor = None
    
    # Determine adapter path from checkpoint path
    adapter_path = None
    if Path(args.checkpoint_path).is_dir():
        adapter_paths = list(Path(args.checkpoint_path).glob("checkpoint-*"))
        if adapter_paths:
            # Sort by checkpoint number and get the latest
            adapter_paths.sort(key=lambda x: int(x.name.split("-")[1]))
            adapter_path = adapter_paths[-1]
            logger.info(f"Found adapter checkpoint at {adapter_path}")
        else:
            adapter_path = args.checkpoint_path
            logger.info(f"Using adapter path directly: {adapter_path}")
    
    # Attempt to load the model with different strategies
    model = None
    errors = []
    
    # Strategy 1: Try loading with the specified model class
    try:
        model_class = get_model_class(args.base_model_path)
        logger.info(f"Attempting to load model with {model_class.__name__}")
        model = model_class.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=args.trust_remote_code
        )
        logger.info(f"Successfully loaded base model using {model_class.__name__}")
    except Exception as e:
        errors.append(f"Failed with {model_class.__name__}: {str(e)}")
        model = None
    
    # Strategy 2: If Strategy 1 failed, try with AutoModelForCausalLM
    if model is None and model_class != AutoModelForCausalLM:
        try:
            logger.info("Attempting to load model with AutoModelForCausalLM")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=args.trust_remote_code
            )
            logger.info("Successfully loaded base model using AutoModelForCausalLM")
        except Exception as e:
            errors.append(f"Failed with AutoModelForCausalLM: {str(e)}")
            model = None
    
    # Strategy 3: Try loading directly from the adapter path (for models saved with full weights)
    if model is None and adapter_path is not None:
        try:
            logger.info(f"Attempting to load full model directly from adapter path: {adapter_path}")
            model = AutoModelForCausalLM.from_pretrained(
                adapter_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=args.trust_remote_code
            )
            logger.info(f"Successfully loaded full model directly from {adapter_path}")
            # If we loaded the full model, we don't need to apply the adapter
            adapter_path = None
        except Exception as e:
            errors.append(f"Failed loading full model from adapter path: {str(e)}")
    
    # If all strategies failed, raise an error
    if model is None:
        error_msg = "Failed to load model using all available strategies:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Load adapter if it's a PEFT model and we haven't loaded the full model already
    if adapter_path is not None:
        try:
            logger.info(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info(f"Successfully loaded adapter from: {adapter_path}")
        except Exception as e:
            logger.error(f"Error loading adapter: {e}")
            raise
    
    return model, tokenizer, processor

def load_dataset(args):
    """Load and preprocess the validation dataset"""
    logger.info(f"Loading dataset from {args.data_file_path}")
    
    # Load the VLM module
    vlm_module_cls = get_vlm_module(args.base_model_path)
    question_prompt = vlm_module_cls.get_question_template(task_type=args.question_task_template)
    
    # Load the JSONL file
    all_data = []
    with open(args.data_file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'image' in item:
                if isinstance(item['image'], str):
                    item['image_path'] = os.path.join(args.image_folder, item['image'])
                    del item['image']
                elif isinstance(item['image'], list):
                    item['image_path'] = [os.path.join(args.image_folder, image) for image in item['image']]
                    del item['image']
                else:
                    raise ValueError(f"Unsupported image type: {type(item['image'])}")
            
            # Extract question and answer
            item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
            item['solution'] = item['conversations'][1]['value']
            del item['conversations']
            
            # Add segmentation path if specified
            if args.segmentation_folder is not None:
                if 'segmentation_path' in item:
                    segmentation_path = os.path.join(args.segmentation_folder, item['segmentation_path'])
                    if os.path.exists(segmentation_path):
                        item['segmentation_path'] = segmentation_path
            
            all_data.append(item)
    
    dataset = Dataset.from_list(all_data)
    
    def make_conversation_from_jsonl(example):
        if 'image_path' in example and example['image_path'] is not None:
            if isinstance(example['image_path'], list):
                return {
                    'image_path': example['image_path'],
                    'problem': example['problem'],
                    'solution': example['solution'],
                    'prompt': [{
                        'role': 'user',
                        'content': [
                            *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                            {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                        ]
                    }],
                    'id': example.get('id', None),
                    'segmentation_path': example.get('segmentation_path', None),
                }
            else:
                return {
                    'image_path': [example['image_path']],
                    'problem': example['problem'],
                    'solution': example['solution'],
                    'prompt': [{
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'text': None},
                            {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                        ]
                    }],
                    'id': example.get('id', None),
                    'segmentation_path': example.get('segmentation_path', None),
                }
        else:
            return {
                'problem': example['problem'],
                'solution': example['solution'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }],
                'id': example.get('id', None),
                'segmentation_path': example.get('segmentation_path', None),
            }
    
    dataset = dataset.map(make_conversation_from_jsonl)
    return dataset

def extract_answer(text):
    """Extract answer from generated text"""
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        return answer_matches[-1].strip()
    else:
        # If no <answer> tag, return the whole text
        return text.strip()

def calculate_metrics(predictions, ground_truths, metrics_list):
    """Calculate evaluation metrics"""
    results = {}
    
    # Format check
    if "format" in metrics_list:
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        format_ok = [bool(re.fullmatch(pattern, pred, re.DOTALL)) for pred in predictions]
        results["format_adherence"] = sum(format_ok) / len(format_ok) if format_ok else 0
    
    # Accuracy check
    if "accuracy" in metrics_list:
        extracted_preds = [extract_answer(pred) for pred in predictions]
        extracted_truths = [extract_answer(gt) for gt in ground_truths]
        
        exact_matches = [pred.strip() == truth.strip() for pred, truth in zip(extracted_preds, extracted_truths)]
        results["exact_match_accuracy"] = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    
    # Add additional metrics from the counting_rewards module
    for metric in metrics_list:
        if metric not in ["format", "accuracy"] and metric in COUNTING_REWARD_FUNCS:
            reward_func = COUNTING_REWARD_FUNCS[metric]
            metric_values = []
            for pred, gt in zip(predictions, ground_truths):
                try:
                    # Wrap predictions for the reward function format
                    completions = [[{"content": pred}]]
                    solution = [gt]
                    kwargs = {}
                    reward = reward_func(completions, solution, **kwargs)[0]
                    metric_values.append(reward)
                except Exception as e:
                    logger.warning(f"Error calculating {metric}: {e}")
                    metric_values.append(0.0)
            
            if metric_values:
                results[metric] = sum(metric_values) / len(metric_values)
    
    return results

def generate_predictions(model, tokenizer, processor, dataset, args):
    """Generate predictions for the validation dataset"""
    model.eval()
    predictions = []
    ground_truths = []
    ids = []
    
    # Process images and generate predictions
    vlm_module_cls = get_vlm_module(args.base_model_path)
    vlm_module = vlm_module_cls()
    
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i:i+args.batch_size]
        batch_preds = []
        
        for item in batch:
            try:
                # Process image and prompt
                model_inputs = vlm_module.prepare_inputs(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    prompt=item['prompt'],
                    image_paths=item['image_path'] if isinstance(item['image_path'], list) else [item['image_path']],
                    max_pixels=args.max_pixels,
                    min_pixels=args.min_pixels
                )
                
                # Generate prediction
                with torch.no_grad():
                    output = model.generate(
                        **model_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        top_p=args.top_p,
                        temperature=args.temperature
                    )
                
                # Decode the output
                prompt_len = model_inputs["input_ids"].shape[1]
                generated_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
                batch_preds.append(generated_text)
                
                # Save ground truth and ID
                ground_truths.append(item['solution'])
                ids.append(item.get('id', i))
                
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                batch_preds.append("")
                ground_truths.append(item['solution'])
                ids.append(item.get('id', i))
        
        predictions.extend(batch_preds)
    
    return predictions, ground_truths, ids

def main():
    parser = HfArgumentParser(EvalArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Parse metrics from comma-separated string
    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    logger.info(f"Evaluating with metrics: {metrics_list}")
    
    # Load model and tokenizer
    model, tokenizer, processor = load_model_and_processor(args)
    
    # Load dataset
    dataset = load_dataset(args)
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions, ground_truths, ids = generate_predictions(model, tokenizer, processor, dataset, args)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    results = calculate_metrics(predictions, ground_truths, metrics_list)
    
    # Print results
    logger.info(f"Evaluation results for {args.checkpoint_path}:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save results to file if specified
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output_file, 'w') as f:
            json.dump({
                "checkpoint_path": args.checkpoint_path,
                "metrics": results,
                "predictions": [{"id": id, "prediction": pred, "ground_truth": gt} 
                                for id, pred, gt in zip(ids, predictions, ground_truths)]
            }, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
