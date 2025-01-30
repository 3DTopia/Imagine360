import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import cv2
import copy
from tqdm import tqdm
from numpy import pi, exp, sqrt
import torchvision.transforms as transforms


from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from animatediff.utils.util import get_boundingbox, replace_video

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDPMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel

from PIL import Image
import PIL

from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from ..utils.util import preprocess_image

from ..utils.video_mask import video_mask

from src.models.MVGenModel import MultiViewBaseModel
from src.utils.Perspective_and_Equirectangular import e2p
from src.utils.pano import pad_pano, unpad_pano
from src.modules.utils import flush, check_cuda_memo


logger = logging.get_logger(__name__) 


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        pers_unet: UNet3DConditionModel,
        pano_unet: UNet3DConditionModel,
        mv_base_model:MultiViewBaseModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_encoder = None,
        image_encoder_name = 'CLIP',
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_pers_unet_version_less_0_9_0 = hasattr(pers_unet.config, "_diffusers_version") and version.parse(
            version.parse(pers_unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_pers_unet_sample_size_less_64 = hasattr(pers_unet.config, "sample_size") and pers_unet.config.sample_size < 64
        if is_pers_unet_version_less_0_9_0 and is_pers_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(pers_unet.config)
            new_config["sample_size"] = 64
            pers_unet._internal_dict = FrozenDict(new_config)



        is_pano_unet_version_less_0_9_0 = hasattr(pano_unet.config, "_diffusers_version") and version.parse(
            version.parse(pano_unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_pano_unet_sample_size_less_64 = hasattr(pano_unet.config, "sample_size") and pano_unet.config.sample_size < 64
        if is_pano_unet_version_less_0_9_0 and is_pano_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(pano_unet.config)
            new_config["sample_size"] = 64
            pano_unet._internal_dict = FrozenDict(new_config)



        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            pers_unet=pers_unet,
            pano_unet=pano_unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
        )


        self.mv_base_model = mv_base_model
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_encoder_name = image_encoder_name
        if image_encoder_name == 'SAM':
            from segment_anything import SamPredictor, sam_model_registry
            self.SAMpredictor = SamPredictor(image_encoder)
            self.SAMProcessor = self.SAMpredictor.transform


    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.image_encoder]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
    
    
    
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1].to(dtype=self.vae.dtype)).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )



    def padding_pano(self, pano, padding=4, latent=False):
        if not latent:
            padding *= 8
        return pad_pano(pano, padding=padding)

    def unpadding_pano(self,pano_pad, padding=4, latent=False):
        if not latent:
            padding *= 8
        return unpad_pano(pano_pad, padding=padding)



    def init_noise(self, bs, video_length, equi_h, equi_w, pers_h, pers_w, cameras, device, latents_dtype=torch.float16):
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
        return pano_noise_output.to(dtype=latents_dtype), noises_output.to(dtype=latents_dtype)



    def prepare_latents(self, batch_size, 
                        num_channels_latents, 
                        video_length, height, 
                        width, dtype, device, 
                        generator, latents=None):
        
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        init_latents = None
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)

        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    def prepare_masked_latents_pano(self, video_length, pano_pixel_values_masked, pano_mask):
        decode_chunk_size = 8
        pano_pixel_values_masked = rearrange(pano_pixel_values_masked, "b f c h w -> (b f) c h w", f=video_length)  
        
        frames = []
        for i in range(0, pano_pixel_values_masked.shape[0], decode_chunk_size):
            num_frames_in = pano_pixel_values_masked[i : i + decode_chunk_size].shape[0]
            frame = self.vae.encode(pano_pixel_values_masked[i : i + decode_chunk_size], num_frames_in).latent_dist.sample()
            frames.append(frame)


        pano_latents_masked = torch.cat(frames)
        pano_latents_masked = rearrange(pano_latents_masked, "(b f) c h w -> b c f h w", f=video_length)
        pano_latents_masked = pano_latents_masked * 0.18215


        pano_mask = pano_mask.transpose(2, 1) #[1, 1, 64, 512, 1024] [b,c,f,h,w]
        pano_mask = torch.nn.functional.interpolate(pano_mask, size=(pano_mask.shape[2], pano_latents_masked.shape[-2], pano_latents_masked.shape[-1]))
        pano_mask = pano_mask.to(self.device)


        return pano_latents_masked, pano_mask


    def prepare_masked_latents_pers(self, video_length, pers_pixel_values_masked, pers_masks):
        decode_chunk_size = 8
        views_num = pers_pixel_values_masked.shape[2]
        pers_pixel_values_masked = rearrange(pers_pixel_values_masked, "b f m c h w -> (b f m) c h w", f=video_length, m=views_num) 
        
        pers_frames = []
        for i in range(0, pers_pixel_values_masked.shape[0], decode_chunk_size):
            num_frames_in = pers_pixel_values_masked[i : i + decode_chunk_size].shape[0]
            pers_frame = self.vae.encode(pers_pixel_values_masked[i : i + decode_chunk_size], num_frames_in).latent_dist.sample()
            pers_frames.append(pers_frame)


        pers_latents_masked = torch.cat(pers_frames)
        pers_latents_masked = rearrange(pers_latents_masked, "(b f m) c h w -> b m c f h w", f=video_length, m=views_num) 
        pers_latents_masked = pers_latents_masked * 0.18215

        pers_masks = pers_masks.permute(0,3,1,2,4,5).squeeze(0)      #[1, 64, 20, 512, 1024] [c,f,m,h,w]
        pers_masks = torch.nn.functional.interpolate(pers_masks, size=(views_num, pers_latents_masked.shape[-2], pers_latents_masked.shape[-1])).unsqueeze(3) #[b f m c h w]
        pers_masks = rearrange(pers_masks, "b f m c h w -> b m c f h w")  
        pers_masks = pers_masks.to(self.device)


        return pers_latents_masked, pers_masks


    def _encode_image_prompt(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image_embeds = self.image_encoder(pil_image.to(self.device)).image_embeds
        uncond_image_prompt_embeds = torch.zeros_like(clip_image_embeds)
        return clip_image_embeds, uncond_image_prompt_embeds
    
    def _encode_image_prompt_plus(self, anchor_pixels_values, ip_plus_condition):
        assert anchor_pixels_values.shape[0] == 1, 'Batch size must be one'

        if self.image_encoder_name == 'CLIP':
            pil_images = []
            if ip_plus_condition == 'image':
                pil_images.append(Image.fromarray(np.uint8(((copy.deepcopy(anchor_pixels_values[:, int(anchor_pixels_values.shape[1]/2), :, :, :])+1.0)/2.0*255).squeeze(0).cpu().numpy().transpose(1,2,0)), mode="RGB"))
            elif ip_plus_condition == 'video':
                n_frames = anchor_pixels_values.shape[1]
                for index in range(n_frames):
                    pil_images.append(Image.fromarray(np.uint8(((copy.deepcopy(anchor_pixels_values[:, index, :, :, :])+1.0)/2.0*255).squeeze(0).cpu().numpy().transpose(1,2,0)), mode="RGB"))
            cond_img = self.clip_image_processor(images=pil_images, return_tensors="pt").pixel_values.to(anchor_pixels_values.device)
            if isinstance(cond_img, Image.Image):
                cond_img = [cond_img]
            if ip_plus_condition == 'image':
                image_embeds = self.image_encoder(cond_img.to(self.device), output_hidden_states=True).hidden_states[-2]
                uncond_image_prompt_embeds = self.image_encoder(torch.zeros_like(cond_img).to(self.device), output_hidden_states=True).hidden_states[-2]
            elif ip_plus_condition == 'video':
                image_embeds = self.image_encoder(cond_img.to(self.device), output_hidden_states=True).hidden_states[-2].unsqueeze(0)
                uncond_image_prompt_embeds = self.image_encoder(torch.zeros_like(cond_img).to(self.device), output_hidden_states=True).hidden_states[-2].unsqueeze(0)
        
        elif self.image_encoder_name == 'SAM':
            assert ip_plus_condition == 'video'
            image_array = np.uint8(((anchor_pixels_values+1.0)/2.0*255).squeeze(dim=0).cpu().numpy().transpose(0,2,3,1))
            image_tensors = []
            for image in image_array:
                # resize the long side to 1024
                image_tensors.append(torch.as_tensor(self.SAMProcessor.apply_image(image), device=anchor_pixels_values.device).permute(2, 0, 1).contiguous())
            image_tensors = torch.stack(image_tensors)
            # pad the short side to 1024 and get features
            batch_size = 8
            assert image_tensors.shape[0]%batch_size == 0
            image_embeds = []
            uncond_image_embeds = []
            for i in range(int(image_tensors.shape[0]/batch_size)):
                self.SAMpredictor.set_torch_image(image_tensors[i*batch_size: (i+1)*batch_size], image_tensors[0].shape[:2])
                image_embeds.append(rearrange(self.SAMpredictor.get_image_embedding(), "f c h w-> f (h w) c"))
                uncond_image_embeds.append(torch.zeros_like(image_embeds[-1]))
            image_embeds = torch.cat(image_embeds, dim=0).unsqueeze(0)
            uncond_image_prompt_embeds = torch.cat(uncond_image_embeds, dim=0).unsqueeze(0)
        
        else:
            raise ValueError

        return image_embeds, uncond_image_prompt_embeds
    
    def get_pixel_pad(self, original_size, target_size, length):
        pad_left = int((target_size[1] - original_size[1]) / 2)
        pad_right = pad_left
        pad_up = int((target_size[0] - original_size[0]) / 2)
        pad_down = pad_up
        pixel_pad = torch.tensor([pad_up, pad_down, pad_left, pad_right], dtype=torch.int16)
        return [pixel_pad]*length   

    
    def _gaussian_weights(self, t_tile_length, t_batch_size):
        from numpy import pi, exp, sqrt

        var = 0.01
        midpoint = (t_tile_length - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        t_probs = [
            exp(-(t - midpoint) * (t - midpoint) / (t_tile_length * t_tile_length) / (2 * var)) / sqrt(2 * pi * var) for
            t in range(t_tile_length)]
        weights = torch.tensor(t_probs)
        weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, t_batch_size, 1, 1)
        return weights



    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale_text: float = 7.5,
        guidance_scale_adapter: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        
        ###new add
        latents_dtype = torch.float16,
        video_batch = None,  
        use_outpaint = False, 
        use_ip_plus_cross_attention = False,
        use_fps_condition=False,
        ip_plus_condition = 'image',
        **kwargs,
    ): 
        """
        video_batch['videoid'] = video_path
        video_batch["fps"] = 8
        video_batch['orig_pixel_values'] = orig_pixel_values.unsqueeze(0)

        video_batch["pano_pixel_values"] = pano_pixel_values.unsqueeze(0) #[b f c h w] [1, 16, 3, 256, 512]
        video_batch["pano_mask"] = pano_mask.unsqueeze(0)
        video_batch['video_length'] = video_length
        video_batch['anchor_pixels_values'] = anchor_pixels_values.unsqueeze(0)   #[1 16, 3, 127, 127]
        video_batch['relative_position'] = relative_position.unsqueeze(0)

        video_batch['pers_pixel_values'] = pers_pixel_values.unsqueeze(0)   #[1, 16, 20, 3, 128, 128] [b f m c h w]
        video_batch['pers_masks'] = pers_masks.unsqueeze(0)                   #[1, 16, 20, 1, 128, 128] [b f m c h w]
        video_batch['cameras'] = cameras
        video_batch['pano_H'] = pano_H
        video_batch['pano_W'] = pano_W
        video_batch['pers_size'] = pers_size

        """     
           
        batch_size = 1
        device = self._execution_device

        do_classifier_free_guidance_text = guidance_scale_text > 1.0       # 8
        do_classifier_free_guidance_adapter = guidance_scale_adapter > 1.0 # -1
        
        # orig_pixel_values = video_batch["orig_pixel_values"] #[b f c h w] [1, 16, 3, 256, 512]
        pano_pixel_values = video_batch["pano_pixel_values"] #[b f c h w]
        pano_mask = video_batch["pano_mask"]               #[b f c h w]
        pers_pixel_values = video_batch["pers_pixel_values"] #[b f m c h w]
        pers_masks = video_batch["pers_masks"]                 #[b f m c h w]
        
        anchor_pixels_values = video_batch["anchor_pixels_values"].to(device) #[b f c h w]
        anchor_pixels_values_pers = video_batch["anchor_pixels_values_pers"].to(device)
        relative_position_tensor = video_batch["relative_position"].to(device)
        pitchs_tensor = video_batch["pitchs"].to(device)

        fps = video_batch["fps"]
        cameras = video_batch["cameras"]
        video_length = video_batch["video_length"]
        views_num = pers_pixel_values.shape[2]
        pers_size = video_batch["pers_size"]
        pano_H, pano_W = video_batch["pano_H"], video_batch["pano_W"]

        pano_negative_prompt = [negative_prompt]
        pers_negative_prompt = pano_negative_prompt * views_num
        

            
        pano_prompt = [prompt]
        pers_prompt = pano_prompt * views_num
        
        
        if use_outpaint:
            pano_pixel_values_masked = pano_pixel_values.clone() * (pano_mask < 0.5)
            pers_pixel_values_masked = pers_pixel_values.clone() * (pers_masks < 0.5)
            pano_pixel_values_masked = pano_pixel_values_masked.to(device)
            pers_pixel_values_masked = pers_pixel_values_masked.to(device)
            
    
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Prepare latent variables
        pano_latent, pers_latent = self.init_noise(batch_size, video_length, pano_H//8, pano_W//8, pers_size//8, pers_size//8, cameras, device, latents_dtype)
        #pano_latent: [b c f h w]
        #pers_latent: [b m c f h w]

       
        pano_latents_masked, pano_mask = self.prepare_masked_latents_pano(video_length, pano_pixel_values_masked, pano_mask)   #[b c f h w]
        pers_latents_masked, pers_masks = self.prepare_masked_latents_pers(video_length, pers_pixel_values_masked, pers_masks) #[b m c f h w]
        

        # ----------------------------------------------------------------------
        # Get the text embedding for conditioning
        with torch.no_grad():
            text_embeddings_pano = self._encode_prompt(pano_prompt, device, num_videos_per_prompt, do_classifier_free_guidance_text, pano_negative_prompt)
            text_embeddings_pers = self._encode_prompt(pers_prompt, device, num_videos_per_prompt, do_classifier_free_guidance_text, pers_negative_prompt)
            
            pano_encoder_hidden_states = text_embeddings_pano.clone().to(dtype=latents_dtype)
            pers_encoder_hidden_states = text_embeddings_pers.clone().to(dtype=latents_dtype)

            
        # ----------------------------------------------------------------------


    
        timesteps = self.scheduler.timesteps
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # ============================================================
        
        #####################################_encode_image_prompt_plus##################################
        if use_ip_plus_cross_attention:
            assert anchor_pixels_values.shape[0] == 1, 'Batch size must be one'

            with torch.no_grad():
                if self.image_encoder_name == 'SAM':
                    assert ip_plus_condition == 'video'
                    image_array = np.uint8(((anchor_pixels_values.to(torch.float32)+1.0)/2.0*255).squeeze(dim=0).cpu().numpy().transpose(0,2,3,1))
                    image_tensors = []
                    for image in image_array:
                        # resize the long side to 1024
                        image_tensors.append(torch.as_tensor(self.SAMProcessor.apply_image(image), device=anchor_pixels_values.device).permute(2, 0, 1).contiguous())
                    image_tensors = torch.stack(image_tensors)
                    # pad the short side to 1024 and get features
                    batch_size = 8
                    assert image_tensors.shape[0]%batch_size == 0
                    image_embeds = []
                    uncond_image_embeds = []
                    for i in range(int(image_tensors.shape[0]/batch_size)):
                        self.SAMpredictor.set_torch_image(image_tensors[i*batch_size: (i+1)*batch_size], image_tensors[0].shape[:2])
                        image_embeds.append(rearrange(self.SAMpredictor.get_image_embedding(), "f c h w-> f (h w) c"))
                        uncond_image_embeds.append(torch.zeros_like(image_embeds[-1]))
                    image_embeds = torch.cat(image_embeds, dim=0).unsqueeze(0)                          #[1, 16, 4096, 256]
                    uncond_image_prompt_embeds = torch.cat(uncond_image_embeds, dim=0).unsqueeze(0)     #[1, 16, 4096, 256]
                    
                    reference_images_clip_feat_pano = torch.cat([image_embeds, image_embeds]).to(dtype=latents_dtype)                                  #[2, 16, 4096, 256]
                    
                    
                    ###############################################################################
                    image_array_pers = np.uint8(((anchor_pixels_values_pers.to(torch.float32)+1.0)/2.0*255).squeeze(dim=0).cpu().numpy().transpose(0,2,3,1))
                    image_tensors_pers = []
                    for image in image_array_pers:
                        # resize the long side to 1024
                        image_tensors_pers.append(torch.as_tensor(self.SAMProcessor.apply_image(image), device=anchor_pixels_values_pers.device).permute(2, 0, 1).contiguous())
                    image_tensors_pers = torch.stack(image_tensors_pers)
                    # pad the short side to 1024 and get features
                    batch_size = 8
                    assert image_tensors_pers.shape[0]%batch_size == 0
                    image_embeds_pers = []
                    uncond_image_embeds_pers = []
                    for i in range(int(image_tensors_pers.shape[0]/batch_size)):
                        self.SAMpredictor.set_torch_image(image_tensors_pers[i*batch_size: (i+1)*batch_size], image_tensors_pers[0].shape[:2])
                        image_embeds_pers.append(rearrange(self.SAMpredictor.get_image_embedding(), "f c h w-> f (h w) c"))
                        uncond_image_embeds_pers.append(torch.zeros_like(image_embeds_pers[-1]))
                    image_embeds_pers = torch.cat(image_embeds_pers, dim=0).unsqueeze(0)                          
                    uncond_image_prompt_embeds_pers = torch.cat(uncond_image_embeds_pers, dim=0).unsqueeze(0)   

                    reference_images_clip_feat_pers = torch.cat([image_embeds_pers, image_embeds_pers]).to(dtype=latents_dtype)                                  #[2, 16, 4096, 256]
                    reference_images_clip_feat_pers = reference_images_clip_feat_pers.unsqueeze(1).repeat(1,views_num,1,1,1).to(dtype=latents_dtype)   #[2, 20, 16, 4096, 256]

    
        #################################################################################################
        


        if use_fps_condition:
            fps_tensor_pano = torch.tensor(fps).to(device).unsqueeze(0)                                
            fps_tensor_pers = fps_tensor_pano.unsqueeze(-1).repeat(1,views_num)   #shape/[2,20]
        

        # flush()
        with torch.no_grad():
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    latents_input_pano = torch.cat((pano_latent, pano_mask, pano_latents_masked), dim=1)
                    latents_input_pers = torch.cat((pers_latent, pers_masks, pers_latents_masked), dim=2)
 
               
                    if do_classifier_free_guidance_text:
                        n_relative_position = 2
                        
                        if use_fps_condition:
                            fps_tensor_pano_input = torch.cat([fps_tensor_pano] * 2)
                            fps_tensor_pers_input = torch.cat([fps_tensor_pers] * 2)
                        else:
                            fps_tensor_pano_input=None
                            fps_tensor_pers_input=None
                            

                        latent_model_input_pano = torch.cat([latents_input_pano] * 2)
                        latent_model_input_pers = torch.cat([latents_input_pers] * 2)

                    latent_model_input_pano = self.scheduler.scale_model_input(latent_model_input_pano, t)  #[2, 9, 16, 32, 64]
                    latent_model_input_pers = self.scheduler.scale_model_input(latent_model_input_pers, t)  #[2, 20, 9, 16, 16, 16]

                    if use_ip_plus_cross_attention:
                        # {Hoffset, Woffset, Hanchor, Wanchor, Htarget, Wtarget}
                        relative_position = relative_position_tensor.unsqueeze(0).repeat(pano_latent.size()[0], 1, 1)
                        relative_position = torch.cat([relative_position] * n_relative_position)  if (n_relative_position>0) else relative_position

                        pitchs = pitchs_tensor.unsqueeze(0).repeat(pano_latent.size()[0], 1)
                        pitchs = torch.cat([pitchs] * n_relative_position)  if (n_relative_position>0) else pitchs
                                        
                    else:
                        relative_position = None
                        pitchs = None
                    
                    torch.cuda.empty_cache()
                     
                    
                    model_pred_pers, model_pred_pano = self.mv_base_model(latents=latent_model_input_pers,
                                                                            pano_latent=latent_model_input_pano,
                                                                            timestep=t.unsqueeze(0),
                                                                            prompt_embd=pers_encoder_hidden_states,      #[20, 77, 1024]
                                                                            pano_prompt_embd=pano_encoder_hidden_states, #[1, 77, 1024]
                                                                            cameras=cameras,
                                                                            
                                                                            use_fps_condition=use_fps_condition,
                                                                            use_ip_plus_cross_attention = use_ip_plus_cross_attention,
                                                                            fps_tensor_pano=fps_tensor_pano_input,              #shape/[2]   value/[8, 8]
                                                                            fps_tensor_pers=fps_tensor_pers_input,              #shape/[2,20]
                                                                            reference_images_clip_feat_pano=reference_images_clip_feat_pano,
                                                                            reference_images_clip_feat_pers=reference_images_clip_feat_pers,
                                                                            relative_position_tensor = relative_position, #only on pano branch
                                                                            pitchs_tensor = pitchs
                                                                            )
                    
                    # perform guidance
                    if do_classifier_free_guidance_text:
                        
                        noise_pred_uncond_pano, noise_pred_text_pano = model_pred_pano.chunk(2)
                        noise_pred_pano = noise_pred_uncond_pano + guidance_scale_text * (noise_pred_text_pano - noise_pred_uncond_pano)
    
                        noise_pred_uncond_pers, noise_pred_text_pers = model_pred_pers.chunk(2)
                        noise_pred_pers = noise_pred_uncond_pers + guidance_scale_text * (noise_pred_text_pers - noise_pred_uncond_pers)
    
    
                    # compute the previous noisy sample x_t -> x_t-1
                    pano_latent = self.scheduler.step(noise_pred_pano, t, pano_latent, **extra_step_kwargs).prev_sample #[1, 4, 16, 32, 64]
                    pers_latent = self.scheduler.step(noise_pred_pers, t, pers_latent, **extra_step_kwargs).prev_sample #[1, 20, 4, 16, 16, 16]

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
                                

                    torch.cuda.empty_cache()
                        

        # Post-processing
        pano_latent_pad = self.padding_pano(pano_latent, latent=True)
        video_pad = self.decode_latents(pano_latent_pad)
        video = self.unpadding_pano(video_pad)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
            
