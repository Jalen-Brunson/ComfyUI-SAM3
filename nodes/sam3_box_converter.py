"""
SAM3 Box Format Converter

Converts between different box formats used by SAM3 nodes:
- SAM3 Image/Text Segmentation outputs: out_boxes_xywh [x, y, w, h] or xyxy format
- SAM3 Video Segmentation expects: SAM3_BOXES_PROMPT {"boxes": [[cx, cy, w, h], ...]}

This node bridges the gap, allowing you to:
1. Detect objects with SAM3 Image/Text Segmentation
2. Feed the detected boxes into SAM3 Video Segmentation for tracking
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class SAM3BoxesToVideoPrompt:
    """
    Convert SAM3 image detection boxes to video tracking prompt format.
    
    This allows using SAM3's text detection on a single frame to initialize
    video tracking with precise bounding boxes.
    
    Workflow:
    1. SAM3 Text Segmentation (image) -> detects objects, outputs boxes
    2. This converter -> transforms to video prompt format  
    3. SAM3 Video Segmentation -> tracks the detected objects
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": ("SAM3_BOXES",),  # From SAM3 image/text segmentation
                "image_width": ("INT", {"default": 1920, "min": 1, "max": 8192}),
                "image_height": ("INT", {"default": 1080, "min": 1, "max": 8192}),
            },
            "optional": {
                "scores": ("SAM3_SCORES",),  # Optional confidence scores
                "score_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only include boxes with score above this threshold"}),
                "max_boxes": ("INT", {"default": 10, "min": 1, "max": 50,
                    "tooltip": "Maximum number of boxes to include"}),
                "box_index": ("INT", {"default": -1, "min": -1, "max": 49,
                    "tooltip": "Select specific box by index (-1 for all boxes)"}),
            }
        }
    
    RETURN_TYPES = ("SAM3_BOXES_PROMPT",)
    RETURN_NAMES = ("boxes_prompt",)
    FUNCTION = "convert"
    CATEGORY = "sam3/utils"
    
    def convert(self, boxes, image_width: int, image_height: int,
                scores=None, score_threshold: float = 0.5, 
                max_boxes: int = 10, box_index: int = -1):
        """
        Convert SAM3 detection boxes to video prompt format.
        
        Input formats supported:
        - Tensor [N, 4] with xywh or xyxy coordinates
        - List of [x, y, w, h] or [x1, y1, x2, y2]
        - Dict with 'boxes' key
        
        Output format:
        - {"boxes": [[cx, cy, w, h], ...]} normalized to 0-1
        """
        # Extract boxes from various input formats
        if isinstance(boxes, dict):
            box_list = boxes.get('boxes', boxes.get('out_boxes_xywh', []))
        elif isinstance(boxes, torch.Tensor):
            box_list = boxes.cpu().numpy().tolist()
        elif isinstance(boxes, np.ndarray):
            box_list = boxes.tolist()
        elif isinstance(boxes, list):
            box_list = boxes
        else:
            print(f"[SAM3 BoxConverter] Unknown box format: {type(boxes)}")
            return ({"boxes": []},)
        
        if len(box_list) == 0:
            print("[SAM3 BoxConverter] No boxes to convert")
            return ({"boxes": []},)
        
        # Extract scores if provided
        score_list = None
        if scores is not None:
            if isinstance(scores, torch.Tensor):
                score_list = scores.cpu().numpy().tolist()
            elif isinstance(scores, np.ndarray):
                score_list = scores.tolist()
            elif isinstance(scores, list):
                score_list = scores
        
        # Detect box format (xywh vs xyxy) by checking if w,h are reasonable
        # xywh: [x, y, width, height] where width/height are positive and smaller than x2-x1
        # xyxy: [x1, y1, x2, y2] where x2 > x1 and y2 > y1
        sample_box = box_list[0]
        is_xyxy = self._detect_xyxy_format(sample_box, image_width, image_height)
        
        print(f"[SAM3 BoxConverter] Detected format: {'xyxy' if is_xyxy else 'xywh'}")
        print(f"[SAM3 BoxConverter] Input boxes: {len(box_list)}, image size: {image_width}x{image_height}")
        
        # Convert and filter boxes
        converted_boxes = []
        for i, box in enumerate(box_list):
            # Filter by index if specified
            if box_index >= 0 and i != box_index:
                continue
            
            # Filter by score if available
            if score_list is not None and i < len(score_list):
                if score_list[i] < score_threshold:
                    continue
            
            # Convert to center format and normalize
            if is_xyxy:
                x1, y1, x2, y2 = box[:4]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
            else:
                x, y, w, h = box[:4]
                cx = x + w / 2
                cy = y + h / 2
            
            # Normalize to 0-1
            cx_norm = cx / image_width
            cy_norm = cy / image_height
            w_norm = w / image_width
            h_norm = h / image_height
            
            # Validate
            if w_norm > 0 and h_norm > 0 and 0 <= cx_norm <= 1 and 0 <= cy_norm <= 1:
                converted_boxes.append([cx_norm, cy_norm, w_norm, h_norm])
                if len(converted_boxes) >= max_boxes:
                    break
        
        print(f"[SAM3 BoxConverter] Converted {len(converted_boxes)} boxes to video prompt format")
        for i, box in enumerate(converted_boxes):
            print(f"  Box {i}: center=({box[0]:.3f}, {box[1]:.3f}), size=({box[2]:.3f}, {box[3]:.3f})")
        
        return ({"boxes": converted_boxes},)
    
    def _detect_xyxy_format(self, box, img_w, img_h) -> bool:
        """Detect if box is in xyxy or xywh format."""
        if len(box) < 4:
            return False
        
        v0, v1, v2, v3 = box[:4]
        
        # If values are normalized (0-1), check if v2 > v0 and v3 > v1 (xyxy pattern)
        if max(v0, v1, v2, v3) <= 1.0:
            # Normalized coordinates
            if v2 > v0 and v3 > v1 and v2 - v0 < 1 and v3 - v1 < 1:
                return True  # Likely xyxy
            return False  # Likely xywh
        
        # Pixel coordinates - check if it's xyxy (x2 > x1, y2 > y1)
        # In xywh, v2 (width) is typically much smaller than v0 (x position)
        # In xyxy, v2 (x2) is larger than v0 (x1)
        if v2 > v0 and v3 > v1:
            # Could be xyxy, but check if w/h are reasonable
            potential_w = v2 - v0
            potential_h = v3 - v1
            # If the "width" and "height" derived from xyxy are reasonable fractions of image
            if potential_w < img_w and potential_h < img_h:
                return True
        
        return False


class SAM3VideoPromptToBoxes:
    """
    Convert SAM3 video prompt format back to regular boxes.
    
    Useful for visualization or further processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes_prompt": ("SAM3_BOXES_PROMPT",),
                "image_width": ("INT", {"default": 1920, "min": 1, "max": 8192}),
                "image_height": ("INT", {"default": 1080, "min": 1, "max": 8192}),
            },
            "optional": {
                "output_format": (["xywh", "xyxy"], {"default": "xyxy"}),
            }
        }
    
    RETURN_TYPES = ("SAM3_BOXES",)
    RETURN_NAMES = ("boxes",)
    FUNCTION = "convert"
    CATEGORY = "sam3/utils"
    
    def convert(self, boxes_prompt: dict, image_width: int, image_height: int,
                output_format: str = "xyxy"):
        """Convert video prompt boxes back to pixel coordinates."""
        
        box_list = boxes_prompt.get("boxes", [])
        if len(box_list) == 0:
            return ([],)
        
        converted = []
        for box in box_list:
            cx, cy, w, h = box
            
            # Denormalize
            cx_px = cx * image_width
            cy_px = cy * image_height
            w_px = w * image_width
            h_px = h * image_height
            
            if output_format == "xyxy":
                x1 = cx_px - w_px / 2
                y1 = cy_px - h_px / 2
                x2 = cx_px + w_px / 2
                y2 = cy_px + h_px / 2
                converted.append([x1, y1, x2, y2])
            else:  # xywh
                x = cx_px - w_px / 2
                y = cy_px - h_px / 2
                converted.append([x, y, w_px, h_px])
        
        return (converted,)


class SAM3MergeBoxPrompts:
    """
    Merge multiple box prompts into one.
    
    Useful for combining boxes from multiple SAM3 text detections.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes_prompt_1": ("SAM3_BOXES_PROMPT",),
            },
            "optional": {
                "boxes_prompt_2": ("SAM3_BOXES_PROMPT",),
                "boxes_prompt_3": ("SAM3_BOXES_PROMPT",),
                "boxes_prompt_4": ("SAM3_BOXES_PROMPT",),
            }
        }
    
    RETURN_TYPES = ("SAM3_BOXES_PROMPT",)
    RETURN_NAMES = ("merged_boxes",)
    FUNCTION = "merge"
    CATEGORY = "sam3/utils"
    
    def merge(self, boxes_prompt_1, boxes_prompt_2=None, boxes_prompt_3=None, boxes_prompt_4=None):
        """Merge multiple box prompts into one."""
        all_boxes = list(boxes_prompt_1.get("boxes", []))
        
        for prompt in [boxes_prompt_2, boxes_prompt_3, boxes_prompt_4]:
            if prompt is not None:
                all_boxes.extend(prompt.get("boxes", []))
        
        print(f"[SAM3 MergeBoxPrompts] Merged {len(all_boxes)} total boxes")
        return ({"boxes": all_boxes},)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3BoxesToVideoPrompt": SAM3BoxesToVideoPrompt,
    "SAM3VideoPromptToBoxes": SAM3VideoPromptToBoxes,
    "SAM3MergeBoxPrompts": SAM3MergeBoxPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3BoxesToVideoPrompt": "SAM3 Boxes → Video Prompt",
    "SAM3VideoPromptToBoxes": "SAM3 Video Prompt → Boxes",
    "SAM3MergeBoxPrompts": "SAM3 Merge Box Prompts",
}
