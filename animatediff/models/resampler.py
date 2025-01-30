# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py

import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm
import torch.nn.functional as F


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
        query_relative_postions = None, 
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.use_relative_postions = query_relative_postions

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, relative_postion_tensor=None):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        if self.use_relative_postions is not None:
            if self.use_relative_postions == 'QuerySum':
                latents = latents + relative_postion_tensor
            elif self.use_relative_postions == 'QueryToken':
                latents = torch.cat((latents, relative_postion_tensor), dim=1)
            else: 
                assert ValueError

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)

class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalProjection(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, compress_video_features=False, kernel_size=4):
        super().__init__()
        self.compress_video_features = compress_video_features
        self.spacial_compress = True if dim < 1024 else False # is SAM feature
        if self.spacial_compress:
            self.patch_embed = nn.Conv2d(in_channels=dim, out_channels=dim*4, kernel_size=4, stride=4, bias=True)
        self.attn_temp = CrossAttention(
            query_dim=dim*4 if self.spacial_compress else dim,
            heads=heads,
            dim_head=dim_head,
            dropout=0.,
            bias=False,
            upcast_attention=False,
        )
        self.norm_temp = nn.LayerNorm(dim*4 if self.spacial_compress else dim)

        self.ff = FeedForward(dim=dim*4 if self.spacial_compress else dim, mult=4)
        self.norm1 = nn.LayerNorm(dim*4 if self.spacial_compress else dim)
        self.kernel_size = kernel_size

        if compress_video_features:
            self.attn_temp_2 = CrossAttention(
            query_dim=dim*4 if self.spacial_compress else dim,
            heads=heads,
            dim_head=dim_head,
            dropout=0.,
            bias=False,
            upcast_attention=False,
        )
            self.norm_temp_2 = nn.LayerNorm(dim*4 if self.spacial_compress else dim)

            self.ff_2 = FeedForward(dim=dim*4 if self.spacial_compress else dim, mult=4)
            self.norm2 = nn.LayerNorm(dim*4 if self.spacial_compress else dim)



    def forward(self, video_hidden_states):
        b, f, d, c = video_hidden_states.shape
        # print('debug --------------', video_hidden_states.shape, self.spacial_compress)
        if self.spacial_compress:
            video_hidden_states = rearrange(video_hidden_states, "b f (h w) c -> (b f) c h w", h=int(math.sqrt(d)))
            video_hidden_states = self.patch_embed(video_hidden_states)
            video_hidden_states = rearrange(video_hidden_states, "(b f) c h w -> b f (h w) c", b=b)
            b, f, d, c = video_hidden_states.shape
            
        video_hidden_states = rearrange(video_hidden_states, "b f d c -> (b d) f c")
        norm_hidden_states = self.norm_temp(video_hidden_states)
        video_hidden_states = self.attn_temp(norm_hidden_states) + video_hidden_states
        video_hidden_states = rearrange(video_hidden_states, "(b d) f c -> b f d c", d=d)

        video_hidden_states = self.ff(self.norm1(video_hidden_states)) + video_hidden_states

        if self.compress_video_features:

            # pooling 1
            video_hidden_states = rearrange(video_hidden_states, "b f d c -> (b d) c f")
            video_hidden_states = F.avg_pool1d(video_hidden_states, kernel_size=self.kernel_size)
            video_hidden_states = rearrange(video_hidden_states, "(b d) c f -> b f d c", d=d)

            # attention 2
            video_hidden_states = rearrange(video_hidden_states, "b f d c -> (b d) f c")
            norm_hidden_states = self.norm_temp_2(video_hidden_states)
            video_hidden_states = self.attn_temp_2(norm_hidden_states) + video_hidden_states
            video_hidden_states = rearrange(video_hidden_states, "(b d) f c -> b f d c", d=d)

            video_hidden_states = self.ff_2(self.norm2(video_hidden_states)) + video_hidden_states

            # pooling 2
            video_hidden_states = rearrange(video_hidden_states, "b f d c -> (b d) c f")
            video_hidden_states = F.avg_pool1d(video_hidden_states, kernel_size=self.kernel_size)
            video_hidden_states = rearrange(video_hidden_states, "(b d) c f -> b f d c", d=d)

        return video_hidden_states