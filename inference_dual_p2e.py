import os
import cv2
import random
from loguru import logger 
import inspect
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from typing import Dict, Tuple
from PIL import Image
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler

from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPTextModel, CLIPTokenizer,AutoModelForCausalLM, AutoTokenizer
import torchvision.transforms as transforms

from animatediff.models.unet import UNet3DConditionModel
from animatediff.utils.util import zero_rank_print


from src.models.MVGenModel import MultiViewBaseModel
from animatediff.pipelines.pipeline_animation_inference_dual import AnimationPipeline

from src.utils.pano import pad_pano, unpad_pano
from src.utils.Perspective_and_Equirectangular import e2p
import src.utils.pano_utils.Equirec2Perspec as E2P
import src.utils.pano_utils.Perspec2Equirec as P2E
from src.utils.pano import get_K_R, icosahedron_sample_camera
from animatediff.utils.video_mask import get_anchor_target
from decord import VideoReader
from sklearn.linear_model import LinearRegression

from animatediff.utils.util import save_videos_grid
from src.modules.utils import flush
from geocalib import GeoCalib

def init_logger():
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)


def padding_pano(pano, padding=8, latent=False):
    if not latent:
        padding *= 8
    return pad_pano(pano, padding=padding)

def unpadding_pano(pano_pad, padding=8, latent=False):
    if not latent:
        padding *= 8
    return unpad_pano(pano_pad, padding=padding)


def get_prompt(frame, lmm_tokenizer, lmm_model):
    path = 'infer_temp.jpg'
    frame = cv2.cvtColor((frame.numpy().transpose(1,2,0)+1)/2*255, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, frame)

    lmm_prompt = "Describe the foreground and possible background of the image content in two sentences. Answer starts with 'The image shows'."

    query = lmm_tokenizer.from_list_format([{'image': path},{'text': lmm_prompt}])
    prompt, _ = lmm_model.chat(lmm_tokenizer, query=query, history=None)
    prompt = prompt.replace('The image shows', '').strip()
    # print(f'=> Get new prompt: {prompt}')
    
    return prompt


def get_cameras(fov=90, pers_resolution=512, device='cuda'):
    thetas, phis = icosahedron_sample_camera()
    thetas, phis = np.rad2deg(thetas), np.rad2deg(phis)

    Ks, Rs = [], []
    for t, p in zip(thetas, phis):
        K, R = get_K_R(fov, t, p,
                        pers_resolution, pers_resolution)  #perspective image size = 256/512
        Ks.append(K)
        Rs.append(R)
    K = np.stack(Ks).astype(np.float32)
    R = np.stack(Rs).astype(np.float32)

    
    cameras = {
        'height': np.full_like(thetas, pers_resolution, dtype=int),   #[20,]
        'width': np.full_like(thetas, pers_resolution, dtype=int),    #[20,]
        'FoV': np.full_like(thetas, fov, dtype=int),                  #[20,]
        'theta': thetas,                                              #[20,]
        'phi': phis,                                                  #[20,]
        'R': R,                                                       #[20,3,3]
        'K': K,                                                       #[20,3,3]
    }
    cameras['height'] = torch.from_numpy(cameras['height']).unsqueeze(0).to(device) #[20,]
    cameras['width'] = torch.from_numpy(cameras['width']).unsqueeze(0).to(device)   #[20,]
    cameras['FoV'] = torch.from_numpy(cameras['FoV']).unsqueeze(0).to(device)       #[20,]
    cameras['theta'] = torch.from_numpy(cameras['theta']).unsqueeze(0).to(device)   #[20,]
    cameras['phi'] = torch.from_numpy(cameras['phi']).unsqueeze(0).to(device)       #[20, 3, 3]
    cameras['R'] = torch.from_numpy(cameras['R']).unsqueeze(0).to(device)           #[20, 3, 3]
    cameras['K'] = torch.from_numpy(cameras['K']).unsqueeze(0).to(device)           #[20, 3, 3]

    return cameras


def process_equi(panovid_data, thetas, phis, pers_resolution=256, back_norm=True):

    #panovid_data [f,c,h,w] torch.float32
    #persvid_data [f,m,c,h,w] (m views)
    thetas = thetas.squeeze()
    phis = phis.squeeze()
    persvid_data = []
    if back_norm:
        panovid_data = (panovid_data + 1) * 127.5           #(-1,1) -> (0,255)
    else:
        panovid_data = panovid_data * 255
    f, c, pano_h, pano_w = panovid_data.shape
    
    for i in range(f):
        pano = panovid_data[i].permute(1,2,0).numpy().astype(np.uint8)
        pers_imgs = []
        equ = E2P.Equirectangular(pano)

        for th, ph in zip(thetas, phis):
            img = equ.GetPerspective(90, th, ph, pers_resolution, pers_resolution)
            pers_img = (img.astype(np.float32)/127.5)-1 if back_norm else np.expand_dims(np.any(img > 0, axis=-1), axis=-1)
            pers_imgs.append(pers_img)
        pers_imgs = np.stack(pers_imgs) #numpy, uint8, [m, h, w, c], (0-1)
        
        persvid_data.append(pers_imgs)

    persvid_data = torch.from_numpy(np.stack(persvid_data, axis=0)).float()

    persvid_data = rearrange(persvid_data,'f m h w c -> f m c h w')
    

    return persvid_data


def init_noise(bs, video_length, equi_h, equi_w, pers_h, pers_w, cameras, device):
    cameras = {k: rearrange(v, 'b m ... -> (b m) ...') for k, v in cameras.items()}

    pano_noise = torch.randn(
        bs, video_length, 1, 4, equi_h, equi_w, device=device)

    pano_noise_output = rearrange(pano_noise.squeeze(2),'b f c h w -> b c f h w')    #[b c f h w]
    pano_noises = pano_noise.expand(-1, -1, len(cameras['FoV']), -1, -1, -1)         #[b f m c h w]
    pano_noises = rearrange(pano_noises, 'b f m c h w -> f b m c h w')               #[f b m c h w]

    noises=[]
    for i in range(video_length):
        pano_noise = pano_noises[i] #[b m c h w]
        pano_noise = rearrange(pano_noise, 'b m c h w -> (b m) c h w')       ##[20, 4, 64, 128]

        noise = e2p(
            pano_noise,
            cameras['FoV'], cameras['theta'], cameras['phi'],
            (pers_h, pers_w), mode='nearest')                   #[(b m) c h w]
        noise = rearrange(noise, '(b m) c h w -> b m c h w', b=bs, m=len(cameras['FoV']))
        noises.append(noise)
        
    noises = torch.stack(noises, dim=0)  #[f b m c h w]
    noises_output = rearrange(noises, 'f b m c h w -> b m c f h w', b=bs, f=video_length, m=len(cameras['FoV']))
    # noise_sample = noise[0, 0, :3]
    # pano_noise_sample = pano_noise[0, 0, :3]
    return pano_noise_output, noises_output

def unet_load_diffusers_lora(unet, state_dict, alpha=1.0, latents_dtype=torch.float16):
    # directly update weight in diffusers model
    for key in state_dict:
        # only process lora down key
        if "up." in key: continue
        up_key    = key.replace(".down.", ".up.")
        model_key = key.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")
        model_key = model_key.replace("to_out.", "to_out.0.")
        layer_infos = model_key.split(".")[:-1]

        curr_layer = unet
        while len(layer_infos) > 0:
            temp_name = layer_infos.pop(0)
            curr_layer = curr_layer.__getattr__(temp_name)
        
        weight_down = state_dict[key]
        weight_up   = state_dict[up_key]
        
        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).to(dtype=latents_dtype).to(curr_layer.weight.data.device)

    return unet

def load_unetbranch(lora_path,
                    lora_alpha, 
                    output_dir, 
                    latents_dtype,
                    enable_xformers_memory_efficient_attention,
                    unet_pretrained_model_path,
                    pretrained_model_path, 
                    unet_additional_kwargs):

    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder="unet", 
        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
    )
    


    if unet_pretrained_model_path!="":
        logging.info(f"from motion pretrained checkpoint: {unet_pretrained_model_path}")
    
        # motion model keys: 'epoch', 'global_step', 'state_dict'
        unet_pretrained_model_path = torch.load(unet_pretrained_model_path, map_location="cpu")

        if "global_step" in unet_pretrained_model_path: zero_rank_print(f"global_step: {unet_pretrained_model_path['global_step']}")
        state_dict = unet_pretrained_model_path["state_dict"] if "state_dict" in unet_pretrained_model_path else unet_pretrained_model_path
        new_state_dict = {}
        
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
            
            
            
        m, u = unet.load_state_dict(new_state_dict, strict=False)
        logging.info(f"unet missing keys: {len(m)}, unexpected keys: {len(u)}")
        
        del state_dict, new_state_dict
        

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # add lora
    if lora_path is not None:
        print(f"load motion LoRA from {lora_path}")
        motion_lora_state_dict = torch.load(lora_path, map_location="cpu")
        motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
        motion_lora_state_dict.pop("animatediff_config", "")

        unet = unet_load_diffusers_lora(unet, motion_lora_state_dict, lora_alpha, latents_dtype)

    return unet

def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    """Convert radians to degrees."""
    return rad / torch.pi * 180

def pers2pano_vid(model, modelname, persframes, pano_H=256, pano_W=512,
                  fov=90, th=0, ph=0):
    pano_frames=[]
    pano_mask=[]  
    ph_list = []
    x=[]
    for i in range(persframes.shape[0]):
        if modelname == 'geocalib':
            # load image as tensor in range [0, 1] with shape [C, H, W]
            img = torch.Tensor(persframes[i]/255).permute(2,0,1).cuda()
            results = model.calibrate(img)
            camera, gravity = results["camera"], results["gravity"]
            vfov = rad2deg(camera.vfov)
            roll, pitch = rad2deg(gravity.rp).unbind(-1)
            roll = roll.item()
            pitch = pitch.item()
            ph_list.append(pitch)
            x.append(i)

        elif modelname == 'perspectivefields':
            predictions = model.inference(img_bgr=persframes[i])
            vfov = predictions['pred_vfov'].cpu().numpy()
            pitch = predictions['pred_pitch'].cpu().numpy()
            roll = predictions['pred_roll'].cpu().numpy()
            logger.info(f"[INFO] Camera estimation modules: vfov:{vfov} pitch:{pitch} roll:{roll}")
            ph_list.append(pitch)
            x.append(i)
        else:
            ph_list.append(ph)
    
    x = np.array(x).reshape((-1,1))
    y = np.array(ph_list)
    linear_model = LinearRegression()
    linear_model.fit(x, y)
    fit_phi = linear_model.predict(x)
    ph_list = fit_phi.tolist()

    for i in range(persframes.shape[0]):
        p2e = P2E.Perspective(persframes[i], fov, th, ph_list[i])
        pano_frame, mask = p2e.GetEquirec(pano_H, pano_W)
        
        pano_frames.append(pano_frame.astype(np.uint8))
        mask = 1 - mask
        mask = np.any(mask > 0, axis=-1).astype(np.uint8)
        mask = mask[...,None]
        pano_mask.append(mask)
 

    pano_frames = np.stack(pano_frames, axis=0)   #  (f, h, w, c)  (64, 256, 512, 3)
    pano_mask = np.stack(pano_mask, axis=0)   #  (f, h, w, c)  (64, 256, 512, 3)
    
    return ph_list, pano_frames, pano_mask, pano_frames

@torch.no_grad()
def main(
    name: str,
    output_dir: str,
    video_path: str,
    pretrained_model_path: str,
    pers_unet_pretrained_model_path: str = "",
    pano_unet_pretrained_model_path: str = "",

    mvmodel_pretrained_model_path: str = "",
    perslora_motion_module_path: str = "",
    panolora_motion_module_path: str = "",
    lora_alpha_pano=1,
    lora_alpha_pers=1,
    video_sample_length=16,
    num_inference_steps=50, 
    unet_additional_kwargs: Dict = {},
    noise_scheduler_kwargs = None,
    enable_xformers_memory_efficient_attention: bool = True,
    global_seed: int = 25,
    use_ip_plus_cross_attention: bool=False,
    image_pretrained_model_path: str="",
    use_fps_condition: bool=False,
    use_outpaint=False,
    angle_adapt='geocalib',
    ip_plus_condition = 'image',
    image_encoder_name = 'CLIP',
    pano_H=512,
    pano_W=1024,
    lmm_path = '',
    prompt = None,
    negative_prompt = None,
):
    
    device = 'cuda'
    
    if global_seed < 0:
        global_seed = random.randint(1, 1000000)

    seed = global_seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


    index = 0

    *_, config = inspect.getargvalues(inspect.currentframe())
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    image_encoder=None
    if use_ip_plus_cross_attention:
        if image_encoder_name == 'SAM':
            from segment_anything import SamPredictor, sam_model_registry
            image_encoder = sam_model_registry["vit_b"](checkpoint=image_pretrained_model_path)
            image_encoder.requires_grad_(False)
            image_encoder.to(device)
        else:
            raise ValueError
        print(f'load image encoder: {image_pretrained_model_path}')

    
    latents_dtype = torch.bfloat16
    model_dtype = latents_dtype
    
    # vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.bfloat16)
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    vae.requires_grad_(False)
    
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    

    if angle_adapt=='perspectivefields':
        from src.modules.perspective2d import PerspectiveFields
        version = 'Paramnet-360Cities-edina-centered'
        pose_estimate_model = PerspectiveFields(version).eval().cuda()
    elif angle_adapt=='geocalib':
        pose_estimate_model = GeoCalib().to(device)
    else:
        pose_estimate_model = None
    

    pers_unet = load_unetbranch(perslora_motion_module_path,
                        lora_alpha_pano, 
                        output_dir, 
                        model_dtype,
                        enable_xformers_memory_efficient_attention,
                        pers_unet_pretrained_model_path,
                        pretrained_model_path, 
                        unet_additional_kwargs)

    pers_unet.requires_grad_(False)

    pano_unet = load_unetbranch(panolora_motion_module_path,
                        lora_alpha_pers, 
                        output_dir, 
                        model_dtype,
                        enable_xformers_memory_efficient_attention,
                        pano_unet_pretrained_model_path,
                        pretrained_model_path, 
                        unet_additional_kwargs)


    pano_unet.requires_grad_(False)


    mv_base_model = MultiViewBaseModel(pers_unet, pano_unet, pano_pad=True)

    mv_base_state_dict_new = {}
    mv_base_state_dict = torch.load(mvmodel_pretrained_model_path, map_location="cpu")
    mv_base_state_dict = mv_base_state_dict["state_dict"] 
            
    
    for old_k, param in mv_base_state_dict.items():
        new_k = old_k.replace('module.', '')
        mv_base_state_dict_new[new_k] = mv_base_state_dict[old_k]

    
    m, u = mv_base_model.load_state_dict(mv_base_state_dict_new, strict=False)
    logger.info(f"mv_base_model missing keys: {len(m)}, unexpected keys: {len(u)}\n")
    del mv_base_state_dict_new, mv_base_state_dict
    mv_base_model.requires_grad_(False)

    vae.to(dtype=model_dtype)
    pers_unet.to(dtype=model_dtype)
    pano_unet.to(dtype=model_dtype)
    mv_base_model.to(dtype=model_dtype)


    pixel_transforms = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])


    generator = torch.Generator(device=device)
    # get different seed for each validation round
    generator.manual_seed(global_seed)
    
    # Move models to GPU
    vae.to(device)    
    text_encoder.to(device)    
    mv_base_model.to(device)
    logging.info("***** Loading Pipeline *****")
        
    
    # Validation pipeline
    pipeline = AnimationPipeline(
        pers_unet=pers_unet, 
        pano_unet=pano_unet,
        mv_base_model=mv_base_model,
        vae=vae, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        scheduler=noise_scheduler, 
        image_encoder=image_encoder, 
        image_encoder_name=image_encoder_name,
    ).to(device)
    pipeline.enable_vae_slicing()
    
    video_path_list = []
    ######################################
    if video_path.endswith('mp4'):
        video_path_list.append(video_path)
        logger.info(f"[INFO]Input single video path:{video_path_list}")
    else:  #dir
        video_dir = video_path
        video_path_list_tmp = os.listdir(video_dir)
        for video_name in video_path_list_tmp:
            if video_name.endswith('mp4'):
                video_path_list.append(os.path.join(video_dir, video_name))
            
        logger.info(f"[INFO] Input multi video path:{video_path_list}")

    # Handle the output folder creation

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/input_vid", exist_ok=True)
    os.makedirs(f"{output_dir}/output_vid", exist_ok=True)
    os.makedirs(f"{output_dir}/mask", exist_ok=True)


    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    for video_path in video_path_list:
        
        video_reader = VideoReader(video_path)
        print(len(video_reader))
        ratio = len(video_reader) // video_sample_length
        if len(video_reader) < video_sample_length:
            ratio = 1
            video_sample_length = len(video_reader)
        
        logger.info('[REPORT] Input video is downsampled in temporal dimension by {}'.format(ratio))
        numbers = [i*ratio for i in range(video_sample_length)]

        video_length = video_sample_length #len(video_reader)
        frames = video_reader.get_batch(numbers) 
        frames = frames.asnumpy()

        if os.path.exists(video_path.replace('.mp4','.txt')):                                                                                                                 
            with open(video_path.replace('.mp4','.txt')) as f:                                                                                                                
                prompt = f.readline()                                                                                                                                         
                logger.info(f"[INFO] Input video prompt:{prompt}")    
                prompt_gen_flag = False                                                                                                            
        elif prompt is None:                                                                                                                               
            prompt_gen_flag = True
        else:
            prompt_gen_flag = False        
                                                                                            
                                                                                                                                                                        
        
        pers_size = int(pano_H / 2)
        cameras = get_cameras(fov=90, pers_resolution=pers_size)        
        orig_pixel_values = pixel_transforms(torch.from_numpy(frames.transpose(0,3,1,2)) / 255.) #[f c h w] [16, 3, 256, 512]

        
        pf_phi, pano_frames, pano_mask, _ = pers2pano_vid(pose_estimate_model, angle_adapt, frames, pano_H=pano_H, pano_W=pano_W) #pano_frames:(16, 256, 512, 3), pano_mask:(16, 256, 512, 1)
        pano_pixel_values = pixel_transforms(torch.from_numpy(pano_frames.transpose(0,3,1,2)) / 255.) #[f c h w] [16, 3, 256, 512]
        pano_mask = torch.from_numpy(pano_mask.transpose(0,3,1,2))  
        pano_mask_tosave = pano_mask.permute(1,0,2,3).unsqueeze(1).expand(-1,3,-1,-1,-1)
        pano_frames_tosave = torch.from_numpy(pano_frames.transpose(0,3,1,2)).permute(1,0,2,3) # c f h w

        anchor_pixels_values, _,_,_, relative_position, pitchs = get_anchor_target(pano_pixel_values, ph_list=pf_phi)        
        anchor_pixels_values, relative_position, pitchs = anchor_pixels_values.squeeze(0), relative_position.squeeze(0), pitchs.squeeze(0)
        
        pers_pixel_values = process_equi(pano_pixel_values, cameras["theta"].cpu().numpy(), cameras["phi"].cpu().numpy(), pers_resolution=pers_size)   #[16, 20, 3, 128, 128] [f m c h w]
        pers_masks = process_equi(pano_mask.repeat(1,3,1,1), cameras["theta"].cpu().numpy(), cameras["phi"].cpu().numpy(), pers_resolution=pers_size, back_norm=False) #[16, 20, 1, 128, 128] [f m c h w]    
        
        
        video_batch_val={}
        
        video_batch_val['videoid'] = video_path
        video_batch_val["fps"] = 8
        video_batch_val['anchor_pixels_values_pers'] = orig_pixel_values.unsqueeze(0).to(dtype=latents_dtype)

        video_batch_val["pano_pixel_values"] = pano_pixel_values.unsqueeze(0).to(dtype=latents_dtype) #[b f c h w] [1, 16, 3, 256, 512]
        video_batch_val["pano_mask"] = pano_mask.unsqueeze(0).to(dtype=latents_dtype)
        video_batch_val['video_length'] = video_length
        video_batch_val['anchor_pixels_values'] = anchor_pixels_values.unsqueeze(0).to(dtype=latents_dtype)   #[1 16, 3, 127, 127]
        video_batch_val['relative_position'] = relative_position.to(dtype=latents_dtype)
        video_batch_val['pitchs'] = pitchs.to(dtype=latents_dtype)

        video_batch_val['pers_pixel_values'] = pers_pixel_values.unsqueeze(0).to(dtype=latents_dtype)   #[1, 16, 20, 3, 128, 128] [b f m c h w]
        video_batch_val['pers_masks'] = pers_masks.unsqueeze(0).to(dtype=latents_dtype)                   #[1, 16, 20, 1, 128, 128] [b f m c h w]
        video_batch_val['cameras'] = cameras
        video_batch_val['pano_H'] = pano_H
        video_batch_val['pano_W'] = pano_W
        video_batch_val['pers_size'] = pers_size


        lmm_tokenizer = AutoTokenizer.from_pretrained(lmm_path, trust_remote_code=True)
        lmm_model = AutoModelForCausalLM.from_pretrained(lmm_path, device_map="cuda", trust_remote_code=True).eval()
        lmm_model.requires_grad_(False)
        # Generate input prompt
        if prompt_gen_flag:
            prompt = get_prompt(orig_pixel_values[4, :,:,:], lmm_tokenizer, lmm_model)
            print(f"[INFO] Prompt generate from llm: {prompt}")


        del lmm_tokenizer
        del lmm_model


        flush()

        with torch.no_grad():
            try:
                videos = pipeline(
                        prompt,
                        latents_dtype=latents_dtype,
                        video_batch=video_batch_val,
                        num_inference_steps = num_inference_steps, 
                        use_outpaint=use_outpaint,
                        generator    = generator,
                        use_ip_plus_cross_attention = use_ip_plus_cross_attention,
                        ip_plus_condition = ip_plus_condition,
                        use_fps_condition=use_fps_condition,
                        negative_prompt=negative_prompt,
                    ).videos
            except:
                continue

            save_videos_grid(pano_mask_tosave, f"{output_dir}/mask/{prompt}.mp4") 
            save_videos_grid(orig_pixel_values.permute(1,0,2,3).unsqueeze(0), f"{output_dir}/input_vid/{prompt[:-1]}.mp4", rescale=True) #bcthw
            pano_frames_tosave = pano_frames_tosave / 255
            save_videos_grid(pano_frames_tosave.unsqueeze(0), f"{output_dir}/mask/color_{prompt}.mp4") 
            save_videos_grid(videos, f"{output_dir}/output_vid/{prompt[:-1]}.mp4")
            
            index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, **config)