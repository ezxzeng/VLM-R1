import base64
from io import BytesIO
import os
from PIL import Image
import torch
import re
from typing import List, Dict, Any, Optional, Tuple, Union

def process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """
    Extract and process image and video information from messages.
    
    Args:
        messages: List of messages in the conversation format
        
    Returns:
        Tuple of (image_tensors, video_tensors)
    """
    image_tensors = []
    video_tensors = []
    has_videos = False
    
    for msg in messages:
        for single_msg in msg:
            for content_item in single_msg["content"]:
                if not isinstance(content_item, dict):
                    continue
                    
                if content_item.get("type") == "image" and "image" in content_item:
                    # Handle images
                    image_data = content_item["image"]
                    if isinstance(image_data, str):
                        # Process image path or base64 encoded image
                        if image_data.startswith("data:"):
                            # Base64 encoded image
                            image_data = re.sub(r'data:image/[^;]+;base64,', '', image_data)
                            image = Image.open(BytesIO(base64.b64decode(image_data)))
                        elif image_data.startswith("file://"):
                            # File path
                            file_path = image_data.replace("file://", "")
                            image = Image.open(file_path).convert('RGB')
                        else:
                            # Assume it's a file path without the prefix
                            image = Image.open(image_data).convert('RGB')
                        
                        # Append the image tensor
                        image_tensors.append(image)
                    else:
                        # Already a PIL image
                        image_tensors.append(image_data)
                        
                elif content_item.get("type") == "video" and "video" in content_item:
                    # Handle videos - this is a placeholder
                    # In a real implementation, you'd process the video data here
                    has_videos = True
                    pass
    
    # Return None for videos if no videos were found
    return image_tensors, video_tensors if has_videos else None 