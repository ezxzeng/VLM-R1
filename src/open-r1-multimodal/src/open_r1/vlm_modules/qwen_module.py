from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch

from open_r1.vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "count":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. When thinking, on each line, list a object that you spot as well as its <x, y> coordinate. The final answer should be a number. For example, <think>I spot a person at <404, 104>.\nI spot a person at <480, 188>.\nTherefore, I see 2 people.</think><answer>2</answer>"
            case "count_many_examples_question_last":
                return "First, count each instance you spot and their <x, y> coordinates as part of the thinking process in <think> </think> tags. Then output the final answer as a single number in <answer> </answer> tags. Below are some examples: \n <think>I spot a person at <401, 105>.\nI spot a person at <48, 19>.\nTherefore, I see 2 people.</think><answer>2</answer> \n <think>I spot an apple at <352, 258>.\nI spot an apple at <45, 351>.\nTherefore, I see 2 apples.</think><answer>2</answer> \n <think>I spot a pizza at <220, 421>.\nTherefore, I see 1 pizza.</think>\n<answer>1</answer> \n <think>I spot a cow at <118, 112>.\nI spot a cow at <143, 207>.\nI spot a cow at <356, 41>.\nTherefore, I see 3 cows.</think>\n<answer>3</answer>. \n {Question}."
            case "count_ratio":
                return "{Question}. First, count each instance you spot and their <x, y> coordinates (where both x and y represent the object's width and height ratios relative to the full image dimensions) as part of the thinking process in <think> </think> tags. Then output the final answer as a single number in <answer> </answer> tags."
            case "count_ratio_example":
                return "{Question}. First, count each instance you spot and their <x, y> coordinates (where both x and y represent the object's width and height ratios relative to the full image dimensions) as part of the thinking process in <think> </think> tags. Then output the final answer as a single number in <answer> </answer> tags. For example, <think>I spot a person at <0.404, 0.104>.\nI spot a person at <0.480, 0.188>.\nTherefore, I see 2 people.</think><answer>2</answer>"
            case "count_ratio_many_examples":
                return "{Question}. First, count each instance you spot and their <x, y> coordinates (where both x and y represent the object's width and height ratios relative to the full image dimensions) as part of the thinking process in <think> </think> tags. Then output the final answer as a single number in <answer> </answer> tags. Below are some examples: \n <think>I spot a person at <0.40, 0.10>.\nI spot a person at <0.48, 0.19>.\nTherefore, I see 2 people.</think><answer>2</answer> \n <think>I spot an apple at <0.35, 0.25>.\nI spot an apple at <0.45, 0.35>.\nTherefore, I see 2 apples.</think><answer>2</answer> \n <think>I spot a pizza at <0.20, 0.42>.\nTherefore, I see 1 pizza.</think>\n<answer>1</answer> \n <think>I spot a cow at <0.18, 0.12>.\nI spot a cow at <0.43, 0.07>.\nI spot a cow at <0.56, 0.41>.\nTherefore, I see 3 cows.</think>\n<answer>3</answer>"
            case "count_ratio_many_examples_question_last":
                return "First, count each instance you spot and their <x, y> coordinates (where both x and y represent the object's width and height ratios relative to the full image dimensions) as part of the thinking process in <think> </think> tags. Then output the final answer as a single number in <answer> </answer> tags. Below are some examples: \n <think>I spot a person at <0.40, 0.10>.\nI spot a person at <0.48, 0.19>.\nTherefore, I see 2 people.</think><answer>2</answer> \n <think>I spot an apple at <0.35, 0.25>.\nI spot an apple at <0.45, 0.35>.\nTherefore, I see 2 apples.</think><answer>2</answer> \n <think>I spot a pizza at <0.20, 0.42>.\nTherefore, I see 1 pizza.</think>\n<answer>1</answer> \n <think>I spot a cow at <0.18, 0.12>.\nI spot a cow at <0.43, 0.07>.\nI spot a cow at <0.56, 0.41>.\nTherefore, I see 3 cows.</think>\n<answer>3</answer>. \n {Question}."
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    
    def format_reward(completions, **kwargs):
        pattern = r"<think>.*?</think>\s*<answer>.*?\[.*?{\"bbox_2d\":\s*\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]\s*,\s*\"label\":\s*\".*?\"\s*}.*?\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
        
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards