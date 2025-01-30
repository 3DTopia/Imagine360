# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import json
import pdb

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config


from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)


from diffusers.loaders import UNet2DConditionLoadersMixin



from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
from .resnet import InflatedConv3d, InflatedGroupNorm
from .resampler import Resampler, TemporalProjection
from safetensors.torch import load_model

logger = logging.get_logger(__name__)


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,      
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        
        # Additional
        use_motion_module              = False,
        motion_module_resolutions      = ( 1,2,4,8 ),
        motion_module_mid_block        = False,
        motion_module_decoder_only     = False,
        motion_module_type             = None,
        motion_module_kwargs           = {},
        unet_use_cross_frame_attention = None,
        unet_use_temporal_attention    = None,
        
        # new
        image_hidden_size              = 1280,
        use_ip_plus_cross_attention    = False,
        scale                          = 1.0,
        num_tokens                     = 4,
        use_learnable_scale            = False,
        use_inflated_groupnorm         = False,
        use_fps_condition              = False,
        use_outpaint                   = False,
        use_relative_postions          = False,
        adapter_cross_attention_dim    = 1024,
        image_cross_attention_dim      = 1024,
        ip_plus_condition              = 'image',
        use_adapter_temporal_projection= False,
        compress_video_features        = False,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4
        self.use_relative_postions = use_relative_postions
        self.image_cross_attention_dim = image_cross_attention_dim
        self.ip_plus_condition = ip_plus_condition

        # input
        if use_outpaint:
            self.conv_in = InflatedConv3d(in_channels*2+1, block_out_channels[0], kernel_size=3, padding=(1, 1))
        else:
            self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        
        if use_relative_postions:
            if use_relative_postions == 'WithTime':
                self.add_cond_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
                add_cond_input_dim = block_out_channels[0]
                self.add_cond_embedding = TimestepEmbedding(add_cond_input_dim*6, time_embed_dim)
                
            elif use_relative_postions == 'WithAdapter':
                self.add_cond_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
                add_cond_input_dim = block_out_channels[0]
                self.add_cond_embedding = TimestepEmbedding(add_cond_input_dim*6, image_cross_attention_dim)
                self.cond_rp_proj = nn.Linear(image_cross_attention_dim,  image_cross_attention_dim // 4 * 3, bias=False)
                self.add_cond_embedding2 = TimestepEmbedding(add_cond_input_dim*1, image_cross_attention_dim // 4 * 1)

            elif use_relative_postions[:5] == 'Query':
                self.add_cond_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
                add_cond_input_dim = block_out_channels[0]
                self.add_cond_embedding = TimestepEmbedding(add_cond_input_dim*6, adapter_cross_attention_dim)

            else:
                raise ValueError("relative positons not supported")
        
        if use_fps_condition:
            self.fps_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
            nn.init.zeros_(self.fps_embedding.linear_2.weight)
            nn.init.zeros_(self.fps_embedding.linear_2.bias)
            
        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if use_ip_plus_cross_attention:
            if ip_plus_condition == 'video' and use_adapter_temporal_projection:
                self.temporal_proj = TemporalProjection(dim=image_hidden_size, dim_head=64, heads=8, compress_video_features=compress_video_features)
            else:
                self.temporal_proj = nn.Identity()
            self.image_proj_model = Resampler(
                dim=adapter_cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=num_tokens,
                embedding_dim = image_hidden_size*4 if self.temporal_proj.spacial_compress else image_hidden_size,
                output_dim = image_cross_attention_dim,
                ff_mult=4,
                query_relative_postions = use_relative_postions if use_relative_postions[:5] == 'Query' else None,
            )

            
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2 ** i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                
                use_motion_module=use_motion_module and (res in motion_module_resolutions) and (not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                
                use_ip_plus_cross_attention = use_ip_plus_cross_attention,
                image_cross_attention_dim = image_cross_attention_dim,
                scale=scale,
                num_tokens=num_tokens,
                use_learnable_scale=use_learnable_scale,
                
                use_inflated_groupnorm=use_inflated_groupnorm,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,


                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                
                
                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                
                use_ip_plus_cross_attention = use_ip_plus_cross_attention,
                image_cross_attention_dim = image_cross_attention_dim,
                scale=scale,
                num_tokens=num_tokens,
                use_learnable_scale=use_learnable_scale,
                
                use_inflated_groupnorm=use_inflated_groupnorm,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        
        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                
                use_motion_module=use_motion_module and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                
                use_ip_plus_cross_attention = use_ip_plus_cross_attention,
                image_cross_attention_dim = image_cross_attention_dim,
                scale=scale,
                num_tokens=num_tokens,
                use_learnable_scale=use_learnable_scale,
                
                use_inflated_groupnorm=use_inflated_groupnorm,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
            
        self.conv_act = nn.SiLU()

        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors



    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)


    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)


    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)





    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D, UNetMidBlock3DCrossAttn)):
            module.gradient_checkpointing = value



    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.enable_freeu
    def enable_freeu(self, s1, s2, b1, b2):
        r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)




    def forward(
        self,
        sample: torch.FloatTensor, #latent_model_input
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_ip_plus_cross_attention: bool=False,
        reference_images_clip_feat=None,
        use_fps_condition=False,
        fps_tensor=None,
        relative_position_tensor=None,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        
        # sample: [2, 9, 64, 64, 128]) [b,c,f,h,w]
        # encoder_hidden_states: [2, 77, 1024]
        
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:#false
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
            
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        
        
        if use_fps_condition:
            if not torch.is_tensor(fps_tensor):
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                fps_tensor = torch.tensor([fps_tensor], dtype=dtype, device=sample.device)
            elif len(fps_tensor.shape) == 0:
                fps_tensor = fps_tensor[None].to(sample.device)
            
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        
        timesteps = timesteps.expand(sample.shape[0])
        #timesteps tensor([951, 951], device='cuda:0') torch.int64
        #fps_tensor tensor([8, 8], device='cuda:0') torch.Size([2]) torch.int64
        
        t_emb = self.time_proj(timesteps)
            
        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if relative_position_tensor is not None and self.use_relative_postions == 'WithTime':
            relative_position_tensor = relative_position_tensor.flatten()
            add_cond_emb = self.add_cond_proj(relative_position_tensor)
            add_cond_emb = add_cond_emb.reshape((emb.shape[0], -1))
            add_cond_emb = self.add_cond_embedding(add_cond_emb)
            emb = emb + add_cond_emb
        
        if use_fps_condition==True:
            fps_tensor = fps_tensor.expand(sample.shape[0]).to(t_emb.device)
            fps_emb = self.time_proj(fps_tensor)
            
            fps_emb = fps_emb.to(dtype=self.dtype)
            emb += self.fps_embedding(fps_emb)
        
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        
        num_frames = sample.shape[2]
            
        # pre-process
        # print("sample1",sample.shape) #[2, 9, 64, 64, 128]
        sample = self.conv_in(sample)
        # print("sample2",sample.shape) #[2, 320, 64, 64, 128]
            
        if use_ip_plus_cross_attention:
            if self.ip_plus_condition == 'video':
                # perform temporal self-attention
                #input reference_images_clip_feat： [2, 64, 4096, 256]
                reference_images_clip_feat = self.temporal_proj(reference_images_clip_feat)
                batch_size, f, num_token, dim_feature = reference_images_clip_feat.shape
                # batch size must be one -> validation classier free guidance use batch_size=2
                # batch_size = 1
                reference_images_clip_feat = reference_images_clip_feat.reshape(batch_size, f*num_token, dim_feature)
                #reference_images_clip_feat： [2, 1024, 1024]

            if relative_position_tensor is not None and self.use_relative_postions[:5] == 'Query':
                relative_position_tensor = relative_position_tensor.flatten()
                add_cond_emb = self.add_cond_proj(relative_position_tensor)
                add_cond_emb = add_cond_emb.reshape((emb.shape[0], -1))
                add_cond_emb = self.add_cond_embedding(add_cond_emb).unsqueeze(dim=1)
                ip_tokens = self.image_proj_model(reference_images_clip_feat, add_cond_emb)
            else:
                ip_tokens = self.image_proj_model(reference_images_clip_feat)
            
            if relative_position_tensor is not None and self.use_relative_postions == 'WithAdapter':
                relative_position_tensor = relative_position_tensor.flatten()
                add_cond_emb = self.add_cond_proj(relative_position_tensor)
                add_cond_emb = add_cond_emb.reshape((emb.shape[0], -1))
                add_cond_emb = self.add_cond_embedding(add_cond_emb).unsqueeze(dim=1)
                ip_tokens = ip_tokens + add_cond_emb
            
            if ip_tokens.shape[2] != encoder_hidden_states.shape[2]:
                # pad the ip tokens

                ip_tokens_pad = torch.zeros([ip_tokens.shape[0], ip_tokens.shape[1], encoder_hidden_states.shape[2]]).to(ip_tokens.device)
                ip_tokens_pad[:,:,:self.image_cross_attention_dim] = ip_tokens
                ip_tokens = ip_tokens_pad
                
            #encoder_hidden_states:[]      ip_tokens: [2, 64, 1024]
            encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1) #[2, 141, 1024]
    
        ##################################################3
    
        # down
        down_block_res_samples = (sample,)    #[2, 320, 64, 64, 128]
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                
                
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)

            down_block_res_samples += res_samples

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, 
                    temb=emb, 
                    res_hidden_states_tuple=res_samples, 
                    upsample_size=upsample_size, 
                    encoder_hidden_states=encoder_hidden_states,
                )

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, unet_additional_kwargs=None):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded temporal unet's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
            
        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]

        from diffusers.utils import WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME
        model = cls.from_config(config, **unet_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)

        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        
        state_dict = torch.load(model_file, map_location="cpu")

        ######## add #########
        if 'use_outpaint' in unet_additional_kwargs:
            model.conv_in.weight.data.zero_()
            model.state_dict()['conv_in.weight'][:,:4,:,:] = state_dict['conv_in.weight']
            for k, v in model.state_dict().items():
                if 'conv_in' in k:
                    state_dict.update({k: v})
        #######################
        
        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        
        params = [p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()]
        print(f"### Temporal Module Parameters: {sum(params) / 1e6} M")  #Params:478M
        
        return model
