#!/usr/bin/env python3
# Custom rewards for counting task

import re
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
import os
from datetime import datetime
from pycocotools import mask as maskUtils


def extract_coordinates(text):
    """Extract all (x,y) coordinates from the think section"""
    coord_pattern = r'<\s*(\d+|\d*\.\d+)\s*,\s*(\d+|\d*\.\d+)\s*>'
    return [(int(x), int(y)) for x, y in re.findall(coord_pattern, text)]

def extract_think_section(text):
    """Extract the thinking section from the response"""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return think_match.group(1).strip() if think_match else ""

def extract_answer_section(text):
    """Extract the answer section from the response"""
    answer_match = re.search(r'<answer>\s*(\d+)\s*</answer>', text, re.DOTALL)
    return int(answer_match.group(1).strip()) if answer_match else None

def count_objects_in_thinking(text):
    """Count how many objects were pointed to in the thinking section"""
    coords = extract_coordinates(text)
    return len(coords)


def counting_format_reward(completions, **kwargs):
    """
    Reward function that checks if the completion follows the specific format for counting tasks.
    The expected format includes:
    - A thinking process in <think></think> tags that lists spotted objects with coordinates
    - Each spotted object should follow the pattern "I spot a [object] at <x, y>."
    - A concluding statement that starts with "Therefore, I see [number] [object]"
    - A final answer that is a single number in <answer></answer> tags
    """
    # Pattern for object identification lines
    object_line_pattern = r"I spot a[n]? .+ at <\s*(\d+|\d*\.\d+)\s*,\s*(\d+|\d*\.\d+)\s*>\."
    # Pattern for conclusion line
    conclusion_pattern = r"Therefore,? I see \d+ .+\."
    
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        # Start with 0 reward
        reward = 0.0
        has_object_line = False
        has_conclusion = False
        has_numeric_answer = False
        # Extract thinking section
        thinking_content = extract_think_section(content)
        has_thinking = bool(thinking_content)
        if has_thinking:
            lines = [line.strip() for line in thinking_content.split('\n') if line.strip()]
            
            # Check if there's at least one object identification line
            has_object_line = any(re.match(object_line_pattern, line) for line in lines)
            reward += 0.33 * has_object_line
            # Check if there's a conclusion line
            has_conclusion = any(re.match(conclusion_pattern, line) for line in lines)
            reward += 0.33 * has_conclusion

        # Check answer format
        answer_content = extract_answer_section(content)
        has_numeric_answer = answer_content is not None
        reward += 0.33 * has_numeric_answer

        rewards.append(reward)
        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path.replace(".txt", "_counting_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Counting Format reward -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Reward: {reward}\n")
                f.write(f"Has thinking: {has_thinking}\n")
                f.write(f"Has object line: {has_object_line}\n")
                f.write(f"Has conclusion: {has_conclusion}\n")
                f.write(f"Has numeric answer: {has_numeric_answer}\n")
    return rewards


def points_reward(completions, solution, **kwargs):
    """
    Reward the model for pointing at objects during reasoning.
    
    Args:
        content: The model's response
        sol: The ground truth response
        
    Returns:
        float: A reward between 0 and 1
    """

    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # Extract points from both ground truth and model response
        think_section = extract_think_section(content)
        if not think_section:
            return 0.0
        
        # Count points in the model response
        points_count = count_objects_in_thinking(think_section)
        
        # Extract the expected count from the solution
        gt_think_section = extract_think_section(sol)
        gt_points_count = count_objects_in_thinking(gt_think_section)

        
        # # Calculate ratio of points found (capped at 1.0)
        # ratio = min(1.0, points_count / gt_points_count)
        
        # return ratio

        reward = 1.0 - min(1.0, abs(points_count - gt_points_count) / gt_points_count)
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path.replace(".txt", "_points_reward.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Points reward -------------\n")
                f.write(f"point_count: {points_count}\n")
                f.write(f"gt_point_count: {gt_points_count}\n")
                f.write(f"reward: {reward}\n")
    return rewards

def point_accuracy_reward(completions, solution, **kwargs):
    """
    Reward based on the spatial accuracy of the predicted points using optimal matching.
    Uses Hungarian algorithm to ensure one-to-one matching between points.
    
    Args:
        content: The model's response
        sol: The ground truth response
        
    Returns:
        float: A reward between 0 and 1
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # Extract coordinates from both responses
        model_points = extract_coordinates(extract_think_section(content))
        gt_points = extract_coordinates(extract_think_section(sol))
            
        if not model_points or not gt_points:
            reward = 0.0
            matched_distances = []
        else:
            # Convert to numpy arrays for vectorized operations
            model_points = np.array(model_points)
            gt_points = np.array(gt_points)
            
            # Create cost matrix of pairwise distances
            # Shape: [num_model_points, num_gt_points]
            cost_matrix = np.zeros((len(model_points), len(gt_points)))
            
            # Vectorized distance calculation
            for i, mp in enumerate(model_points):
                # Broadcasting to compute distances to all gt_points at once
                distances = np.sqrt(np.sum((gt_points - mp)**2, axis=1))
                cost_matrix[i] = distances
            
            # Use Hungarian algorithm to find optimal matching
            # (minimizes total distance between matched points)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Get distances for matched pairs
            matched_distances = cost_matrix[row_ind, col_ind]
            
            # Convert distances to rewards (closer = higher reward)
            # Typical image is around 640x480, so max distance ~800px
            max_possible_dist = 800
            _rewards = np.maximum(0, 1 - (matched_distances / max_possible_dist))
            reward = float(np.mean(_rewards)) if len(_rewards) > 0 else 0.0

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path.replace(".txt", "_point_accuracy_reward.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Point accuracy reward -------------\n")
                f.write(f"model_points: {model_points}\n")
                f.write(f"gt_points: {gt_points}\n")
                f.write(f"matched_distances: {matched_distances}\n")
                f.write(f"reward: {reward}\n")
        rewards.append(reward)
        
    # Return average reward across all matched points
    return rewards

def count_consistency_reward(completions, solution, **kwargs):
    """
    Reward the model for consistency between points and final count.
    
    Args:
        content: The model's response
        sol: The ground truth response
        
    Returns:
        float: A reward between 0 and 1
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        think_section = extract_think_section(content)
        if not think_section:
            reward = 0.0
        else:
            # Count the points in the thinking section
            points_count = count_objects_in_thinking(think_section)
                
            # Extract the final count from the response
            final_count = extract_answer_section(content)
            
            # If no final count could be extracted, no reward
            if final_count is None or points_count != final_count:
                reward = 0.0
            else:
                # Check if the number of points matches the final count
                reward = 1.0
        
        rewards.append(reward)
    
    return rewards


def counting_accuracy_reward(completions, solution, **kwargs):
    """
    Reward the model for correctly counting objects.
    
    Args:
        content: The model's response
        sol: The ground truth response
        
    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # Extract the final counts from both model and ground truth
        model_count = extract_answer_section(content)
        gt_count = extract_answer_section(sol)
        
        # If either count couldn't be extracted, no reward
        if model_count is None or gt_count is None:
            rewards.append(0.0)
        else:
            # Check if the counts match
            reward = 1.0 if model_count == gt_count else 0.0
            rewards.append(reward)
    
    return rewards


def segmentation_reward(completions, solution, **kwargs):
    """
    Reward the model for correctly segmenting objects.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, segmentation_filepath in zip(contents, solution, kwargs.get("segmentation_path")):
        gt_segmentation = np.load(segmentation_filepath)

        model_points = extract_coordinates(extract_think_section(content))

        instance_ids_hit = set()

        # check if the model points are in the gt segmentation
        for point in model_points:
            if point[0] < 0 or point[0] >= gt_segmentation.shape[1] or point[1] < 0 or point[1] >= gt_segmentation.shape[0]:
                continue
            instance_id = gt_segmentation[point[1], point[0]]
            if instance_id > 0: #ignore background
                instance_ids_hit.add(instance_id)
        
        reward = len(instance_ids_hit) / gt_segmentation.max()
        rewards.append(reward)

    return rewards



# def combined_counting_reward(completions, solution, **kwargs):
#     """
#     Combined reward function that balances all aspects of the counting task.
    
#     Args:
#         content: The model's response
#         sol: The ground truth response
        
#     Returns:
#         float: A reward between 0 and 1
#     """
#     # Get individual rewards
#     pointing_reward = points_reward(content, sol)
#     point_acc_reward = point_accuracy_reward(content, sol)
#     consistency_reward = count_consistency_reward(content, sol)
#     accuracy_reward = counting_accuracy_reward(content, sol)
#     format_reward = counting_format_reward(content)
    
#     # Combine rewards with weights
#     weights = {
#         'pointing': 0.20,
#         'point_accuracy': 0.25,
#         'consistency': 0.15,
#         'accuracy': 0.30,
#         'format': 0.10
#     }
    
#     combined_reward = (
#         weights['pointing'] * pointing_reward +
#         weights['point_accuracy'] * point_acc_reward +
#         weights['consistency'] * consistency_reward +
#         weights['accuracy'] * accuracy_reward +
#         weights['format'] * format_reward
#     )
    
#     return combined_reward




# Register all reward functions
COUNTING_REWARD_FUNCS = {
    'points': points_reward,
    'point_accuracy': point_accuracy_reward, # pointing at right number of objects
    'count_consistency': count_consistency_reward, # number of points same as final answer
    'counting_accuracy': counting_accuracy_reward, # spacial accuracy of points
    'segmentation_reward': segmentation_reward, # points landing on segmentation masks
    # 'combined_counting': combined_counting_reward,
    'counting_format_reward': counting_format_reward, # format of the response
} 