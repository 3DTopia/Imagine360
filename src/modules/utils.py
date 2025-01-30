import torch
import gc
import cv2
import numpy as np
from einops import rearrange
from PIL import Image


def tensor_to_image(image):
    if image.dtype != torch.uint8:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image * 255).round()
    image = image.cpu().numpy().astype('uint8')
    image = rearrange(image, '... c h w -> ... h w c')
    return image


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def check_cuda_memo(info="", device=0):
    print(f"================= cuda memory info {info} ==================")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    
    free_memory = total_memory - (allocated_memory + cached_memory)
    
    print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
    print(f"Cached memory: {cached_memory / 1024**2:.2f} MB")
    print(f"Free memory: {free_memory / 1024**2:.2f} MB")
    print(f"=====================================================\n")
    
    
    
def get_maxrec_cord(input):
    # input: [h,w] [0-1] (maskinput better) 
    if isinstance(input, torch.Tensor):
        input = input.cpu().numpy()
    height, width = input.shape
    dp = np.zeros((height, width), dtype=int)

    for i in range(height):
        for j in range(width):
            if input[i, j] == 1:
                dp[i, j] = dp[i - 1, j] + 1 if i > 0 else 1

    max_area = 0
    max_rect = (0, 0, 0, 0)  

    for i in range(height):
        stack = []
        for j in range(width + 1):
            h = dp[i, j] if j < width else 0
            while stack and h < dp[i, stack[-1]]:
                height_idx = stack.pop()
                height_val = dp[i, height_idx]
                width_val = j if not stack else j - stack[-1] - 1
                area = height_val * width_val
                if area > max_area:
                    max_area = area
                    max_rect = (i - height_val + 1, stack[-1] + 1 if stack else 0, width_val, height_val)
            stack.append(j)
            
    # top_left_y, top_left_x, rect_width, rect_height = max_rect
    
    #cropped = orig[top_left_y:top_left_y + rect_height, top_left_x:top_left_x + rect_width]

    
    return max_rect