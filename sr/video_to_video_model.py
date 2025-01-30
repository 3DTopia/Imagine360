from typing import Any, Dict

from diffusers import AutoencoderKLTemporalDecoder
from einops import rearrange
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.modules.embedder import FrozenOpenCLIPEmbedder
import video_to_video.modules.unet_v2v as unet_v2v
from video_to_video.utils.config import cfg
from video_to_video.utils.logger import get_logger
from video_to_video.utils.util import *
from utils.pano import pad_pano, unpad_pano


logger = get_logger()

def padding_pano(pano, padding=16, latent=False):
    if not latent:
        padding *= 8
    return pad_pano(pano, padding=padding)

def unpadding_pano(pano_pad, padding=16, latent=False):
    if not latent:
        padding *= 8
    return unpad_pano(pano_pad, padding=padding)


class VideoToVideo:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(f"cuda")
        clip_encoder = FrozenOpenCLIPEmbedder(device=self.device, pretrained="laion2b_s32b_b79k")
        clip_encoder.model.to(self.device)
        self.clip_encoder = clip_encoder
        logger.info(f"Build encoder with {cfg.embedder.type}")

        generator = unet_v2v.ControlledV2VUNet()
        generator = generator.to(self.device)
        generator.eval()

        cfg.model_path = opt.model_path
        load_dict = torch.load(cfg.model_path, map_location="cpu")
        if "state_dict" in load_dict:
            load_dict = load_dict["state_dict"]
        ret = generator.load_state_dict(load_dict, strict=True)

        self.generator = generator.half()
        logger.info("Load model path {}, with local status {}".format(cfg.model_path, ret))

        sigmas = noise_schedule(
            schedule="logsnr_cosine_interp", n=1000, zero_terminal_snr=True, scale_min=2.0, scale_max=4.0
        )
        diffusion = GaussianDiffusion(sigmas=sigmas)
        self.diffusion = diffusion
        logger.info("Build diffusion with GaussianDiffusion")

        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
        )
        vae.to(self.device)
        vae.eval()
        vae.requires_grad_(False)
        self.vae = vae

        torch.cuda.empty_cache()

        self.negative_prompt = cfg.negative_prompt
        self.positive_prompt = cfg.positive_prompt

        negative_y = clip_encoder(self.negative_prompt).detach()
        self.negative_y = negative_y

    def test(
        self,
        input: Dict[str, Any],
        total_noise_levels=1000,
        steps=50,
        solver_mode="fast",
        guide_scale=7.5,
        noise_aug=200,
    ):
        video_data = input["video_data"]
        y = input["y"]
        mask_cond = input["mask_cond"]
        s_cond = input["s_cond"]
        interp_f_num = input["interp_f_num"]


        key_f_num = len(video_data)
        aug_video = []
        for i in range(key_f_num):
            if i == key_f_num - 1:
                aug_video.append(video_data[i : i + 1])
            else:
                aug_video.append(video_data[i : i + 1].repeat(interp_f_num + 1, 1, 1, 1))
        video_data = torch.concat(aug_video, dim=0)

        logger.info(f"video_data shape: {video_data.shape}")
        
        frames_num, _, h, w = video_data.shape
        padding = pad_to_fit(h, w)
        w1, w2, h1, h2 = padding
        video_data = F.pad(video_data, (w1,w2,h1,h2), "circular")
        video_data = video_data.unsqueeze(0)
        bs = 1
        video_data = video_data.to(self.device)

        mask_cond = mask_cond.unsqueeze(0).to(self.device)
        s_cond = torch.LongTensor([s_cond]).to(self.device)

        video_data_feature = self.vae_encode(video_data)
        torch.cuda.empty_cache()

        y = self.clip_encoder(y).detach()

        with amp.autocast(enabled=True):

            t_hint = torch.LongTensor([noise_aug - 1]).to(self.device)
            video_in_low_fps = video_data_feature[:, :, :: interp_f_num + 1].clone()
            noised_hint = self.diffusion.diffuse(video_in_low_fps, t_hint)

            t = torch.LongTensor([total_noise_levels - 1]).to(self.device)
            noised_lr = self.diffusion.diffuse(video_data_feature, t)

            model_kwargs = [{"y": y}, {"y": self.negative_y}]
            model_kwargs.append({"hint": noised_hint})
            model_kwargs.append({"mask_cond": mask_cond})
            model_kwargs.append({"s_cond": s_cond})
            model_kwargs.append({"t_hint": t_hint})

            torch.cuda.empty_cache()
            chunk_inds = make_chunks(frames_num, interp_f_num) #if frames_num > 32 else None

            solver = "dpmpp_2m_sde"  # 'heun' | 'dpmpp_2m_sde'
            gen_vid = self.diffusion.sample(
                noise=noised_lr,
                model=self.generator,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=0.2,
                solver=solver,
                solver_mode=solver_mode,
                steps=steps,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization="trailing",
                chunk_inds=chunk_inds,
            )
            torch.cuda.empty_cache()

            logger.info(f"sampling, finished.")
            gen_vid = gen_vid[:, :,  :, h1 //8 : (h + h1)//8, w1//8 : (w + w1)//8]
            gen_vid_pad = padding_pano(gen_vid, latent=True)
            gen_video = self.tiled_chunked_decode(gen_vid_pad)
            gen_video = unpadding_pano(gen_video)
            logger.info(f"temporal vae decoding, finished.")

        torch.cuda.empty_cache()

        return gen_video.type(torch.float32).cpu()

    def temporal_vae_decode(self, z, num_f):
        return self.vae.decode(z / self.vae.config.scaling_factor, num_frames=num_f).sample

    def vae_encode(self, t, chunk_size=1):
        bs = t.shape[0]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        z_list = []
        for ind in range(0, t.shape[0], chunk_size):
            z_list.append(self.vae.encode(t[ind : ind + chunk_size]).latent_dist.sample())
        z = torch.cat(z_list, dim=0)
        z = rearrange(z, "(b f) c h w -> b c f h w", b=bs)
        return z * self.vae.config.scaling_factor

    def tiled_chunked_decode(self, z):
        batch_size, num_channels, num_frames, height, width = z.shape

        self.frame_chunk_size = 5
        self.tile_img_height = 576
        self.tile_img_width = 1024

        self.tile_overlap_ratio_height = 1 / 4
        self.tile_overlap_ratio_width = 1 / 4
        self.tile_overlap_ratio_time = 1 / 2

        overlap_img_height = int(self.tile_img_height * self.tile_overlap_ratio_height)
        overlap_img_width = int(self.tile_img_width * self.tile_overlap_ratio_width)

        self.tile_z_height = self.tile_img_height // 8
        self.tile_z_width = self.tile_img_width // 8

        overlap_z_height = overlap_img_height // 8
        overlap_z_width = overlap_img_width // 8

        overlap_time = int(self.frame_chunk_size * self.tile_overlap_ratio_time)

        images = z.new_zeros((batch_size, 3, num_frames, height * 8, width * 8))
        count = images.clone()
        height_inds = sliding_windows_1d(height, self.tile_z_height, overlap_z_height)
        for start_height, end_height in height_inds:
            width_inds = sliding_windows_1d(width, self.tile_z_width, overlap_z_width)
            for start_width, end_width in width_inds:
                time_inds = sliding_windows_1d(num_frames, self.frame_chunk_size, overlap_time)
                time = []
                for start_frame, end_frame in time_inds:
                    # logger.info(f'{start_height},{start_width},{start_frame}')
                    tile = z[
                        :,
                        :,
                        start_frame:end_frame,
                        start_height:end_height,
                        start_width:end_width,
                    ]
                    tile_f_num = tile.size(2)
                    tile = rearrange(tile, "b c f h w -> (b f) c h w")
                    tile = self.temporal_vae_decode(tile, tile_f_num)
                    tile = rearrange(tile, "(b f) c h w -> b c f h w", b=batch_size)
                    time.append(tile)
                blended_time = []
                for k, chunk in enumerate(time):
                    if k > 0:
                        chunk = blend_time(time[k - 1], chunk, overlap_time)
                    if k != len(time) - 1:
                        chunk_size = chunk.size(2)
                        blended_time.append(chunk[:, :, : chunk_size - overlap_time])
                    else:
                        blended_time.append(chunk)
                tile_blended_time = torch.cat(blended_time, dim=2)

                _, _, _, tile_h, tile_w = tile_blended_time.shape
                weights = gaussian_weights(tile_w, tile_h)[None, None, None]
                weights = torch.tensor(weights, dtype=images.dtype, device=images.device)

                images[:, :, :, start_height * 8 : end_height * 8, start_width * 8 : end_width * 8] += (
                    tile_blended_time * weights
                )
                count[:, :, :, start_height * 8 : end_height * 8, start_width * 8 : end_width * 8] += weights

        images.div_(count)

        return images
