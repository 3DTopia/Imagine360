import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
import torch.distributed as dist
import PIL
import cv2
import copy

from tqdm import tqdm
from einops import rearrange
import torchvision.transforms as transforms
import torch.nn.functional as F

def save_tensor_as_image(tensor, filename):
    img = transforms.ToPILImage()(tensor)

    img.save(filename, 'PNG')

def clip_image_processor_padding(image_tensors, target_size=224):
    processed_images = []

    for img_tensor in image_tensors:
        _, height, width = img_tensor.shape
        scale_factor = target_size / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

        pad_left = (target_size - new_width) // 2
        pad_top = (target_size - new_height) // 2
        pad_right = target_size - new_width - pad_left
        pad_bottom = target_size - new_height - pad_top

        img_padded = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom))

        processed_images.append(img_padded.squeeze(0))

    stacked_tensor = torch.stack(processed_images)

    return stacked_tensor


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

def save_list_to_txt(lst, output_file):
    with open(output_file, 'w') as f:
        for item in lst:
            f.write(f'{item}\n')

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, n_frames=None):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    index = 0
    for x in videos:
        if n_frames is not None:
            if index == n_frames:
                break
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
        index = index + 1

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def replace_video(video, videos_masked, pad, smooth=True):
    [pad_up, pad_down, pad_left, pad_right] = pad
    _,_,_,height,width = video.shape
    pad_down = height - pad_down
    pad_right = width - pad_right

    video_replace = copy.deepcopy(video)
    videos_masked = rearrange(videos_masked, 'b f c h w -> b c f h w')
    videos_masked = (videos_masked-videos_masked.min())/(videos_masked.max() - videos_masked.min()) 
    videos_masked = videos_masked*(video_replace.max() - video_replace.min()) + video_replace.min()
    video_replace[:, :, :, pad_up:pad_down, pad_left:pad_right] = videos_masked[:, :, :, pad_up:pad_down, pad_left:pad_right]
    h_ori, w_ori = int(pad_down-pad_up), int(pad_right - pad_left)
    video_smooth = video_replace.clone()
    if smooth:
        margin_size = max(max(height, width) // 50, 2)
        margin_sigma = 2
        
        margin_left = margin_size if pad_left != 0 else 1
        margin_right = margin_size if pad_right != 0 else 1
        margin_up = margin_size if pad_up != 0 else 1
        margin_down = margin_size if pad_down != 0 else 1
        
        kernel_w = w_ori + (w_ori % 2 + 1)
        kernel_h = h_ori + (h_ori % 2 + 1)

        sigma_h, sigma_w = margin_sigma, margin_sigma

        mask = np.zeros((w_ori, h_ori, 1))
        mask[margin_left:-margin_right, margin_up:-margin_down, :] = np.ones((w_ori - margin_left - margin_right, h_ori - margin_up - margin_down, 1))
        mask = cv2.GaussianBlur(mask, (kernel_w, kernel_h), sigmaX=sigma_w, sigmaY=sigma_h, borderType=cv2.BORDER_DEFAULT)

        mask = np.transpose(mask)  # h, w
        mask = torch.from_numpy(mask)
        b,c,t,_,_ = video.shape
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, c, t, 1, 1)
        video_smooth[:, :, :, pad_up:pad_down, pad_left:pad_right] = video_replace[:, :, :, pad_up:pad_down, pad_left:pad_right] * mask + video[:, :, :, pad_up:pad_down, pad_left:pad_right] * (1 - mask)
    return video_smooth, video_replace
    

def get_boundingbox(video, pad, thick=2, round=0):

    [pad_up, pad_down, pad_left, pad_right] = pad
    _,_,_,height,width = video.shape
    pad_down = height - pad_down
    pad_right = width - pad_right

    video_split = copy.deepcopy(video)
    
    video_split[:, 0, :, pad_up:(pad_up + thick), pad_left:pad_right] = 1
    video_split[:, 1, :, pad_up:(pad_up + thick), pad_left:pad_right] = 0
    video_split[:, 2, :, pad_up:(pad_up + thick), pad_left:pad_right] = 0

    video_split[:, 0, :, (pad_down - thick):pad_down, pad_left:pad_right] = 1
    video_split[:, 1, :, (pad_down - thick):pad_down, pad_left:pad_right] = 0
    video_split[:, 2, :, (pad_down - thick):pad_down, pad_left:pad_right] = 0

    video_split[:, 0, :, pad_up:pad_down, pad_left:(pad_left + thick)] = 1
    video_split[:, 1, :, pad_up:pad_down, pad_left:(pad_left + thick)] = 0
    video_split[:, 2, :, pad_up:pad_down, pad_left:(pad_left + thick)] = 0

    video_split[:, 0, :, pad_up:pad_down, (pad_right - thick):pad_right] = 1
    video_split[:, 1, :, pad_up:pad_down, (pad_right - thick):pad_right] = 0
    video_split[:, 2, :, pad_up:pad_down, (pad_right - thick):pad_right] = 0

    if round >0:
        for r in range(round):
            [pad_up_2, pad_down_2, pad_left_2, pad_right_2] = pad
            pad_up, pad_down, pad_left, pad_right = pad_up_2+pad_up, pad_down_2+pad_up, pad_left_2+pad_left, pad_right_2+pad_left
            _,_,_,height,width = video.shape
            pad_down = height - pad_down
            pad_right = width - pad_right
            
            video_split[:, 0, :, pad_up:(pad_up + thick), pad_left:pad_right] = 0
            video_split[:, 1, :, pad_up:(pad_up + thick), pad_left:pad_right] = 1
            video_split[:, 2, :, pad_up:(pad_up + thick), pad_left:pad_right] = 0

            video_split[:, 0, :, (pad_down - thick):pad_down, pad_left:pad_right] = 0
            video_split[:, 1, :, (pad_down - thick):pad_down, pad_left:pad_right] = 1
            video_split[:, 2, :, (pad_down - thick):pad_down, pad_left:pad_right] = 0

            video_split[:, 0, :, pad_up:pad_down, pad_left:(pad_left + thick)] = 0
            video_split[:, 1, :, pad_up:pad_down, pad_left:(pad_left + thick)] = 1
            video_split[:, 2, :, pad_up:pad_down, pad_left:(pad_left + thick)] = 0

            video_split[:, 0, :, pad_up:pad_down, (pad_right - thick):pad_right] = 0
            video_split[:, 1, :, pad_up:pad_down, (pad_right - thick):pad_right] = 1
            video_split[:, 2, :, pad_up:pad_down, (pad_right - thick):pad_right] = 0

    return video_split


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents

def load_weights(
    animation_pipeline,
    # motion module
    motion_module_path         = "",
    motion_module_lora_configs = [],
    # image layers
    dreambooth_model_path = "",
    lora_model_path       = "",
    lora_alpha            = 0.8,
):
    # 1.1 motion module
    unet_state_dict = {}
    if motion_module_path != "":
        print(f"load motion module from {motion_module_path}")
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
        unet_state_dict.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
    
    missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
    assert len(unexpected) == 0
    del unet_state_dict

    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
            
        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, animation_pipeline.vae.config)
        animation_pipeline.vae.load_state_dict(converted_vae_checkpoint)
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, animation_pipeline.unet.config)
        animation_pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # 3. text_model
        animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        del dreambooth_state_dict
        
    if lora_model_path != "":
        print(f"load lora model from {lora_model_path}")
        assert lora_model_path.endswith(".safetensors")
        lora_state_dict = {}
        with safe_open(lora_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                lora_state_dict[key] = f.get_tensor(key)
                
        animation_pipeline = convert_lora(animation_pipeline, lora_state_dict, alpha=lora_alpha)
        del lora_state_dict


    for motion_module_lora_config in motion_module_lora_configs:
        path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
        print(f"load motion LoRA from {path}")

        motion_lora_state_dict = torch.load(path, map_location="cpu")
        motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict

        animation_pipeline = convert_motion_lora_ckpt_to_diffusers(animation_pipeline, motion_lora_state_dict, alpha)

    return animation_pipeline

def preprocess_image(image, height, width):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        min_size = min(width, height)
        pixel_transforms = transforms.Compose([
            transforms.Resize(min_size),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        image = pixel_transforms(image[0]).unsqueeze(0)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image