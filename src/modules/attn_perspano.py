import torch
import torch.nn as nn
from .transformer import BasicTransformerBlock, SphericalPE
from src.utils.utils import get_coords, get_merged_masks
from einops import rearrange, repeat
import numpy as np
from src.modules.utils import flush, check_cuda_memo
from PIL import Image

class WarpAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.mv_attn = BasicTransformerBlock(dim, dim//32, 32, context_dim=dim)
        self.pe = SphericalPE(dim//4) 
        self.n_groups = 2
        self.dim = dim



    def forward(self, pers_x, equi_x, cameras):
        
        # e.g 2Dmodule:   pers_x [20, 320, 16, 16]     equi_x:[1, 320, 32, 64]

        # pers_x: [20, 320, 16, 16, 16] (bm,c,f,h,w)    %ori_2D [20, 320, 16, 16] (bm,c,h,w)
        # equi_x: [1,  320, 16, 32, 64] (b,c,f,h,w)    %ori_2D [1, 320, 32, 64]  (b,c,h,w)
        
        # height torch.Size([20])
        # width torch.Size([20])
        # FoV torch.Size([20])
        # theta torch.Size([20])
        # phi torch.Size([20])
        # R torch.Size([20, 3, 3])
        # K torch.Size([20, 3, 3])
        
        bm, c, f, pers_h, pers_w = pers_x.shape
        b,  c, f, equi_h, equi_w = equi_x.shape
        m = bm // b
        pers_masks, equi_masks = get_merged_masks(
            pers_h, pers_w, equi_h, equi_w, cameras, pers_x.device, pers_x.dtype)
        

        pers_masks = pers_masks.unsqueeze(0).expand(b,-1,-1,-1,-1,-1)                     #b m eh ew ph pw
        pers_masks = rearrange(pers_masks, 'b m eh ew ph pw  -> (b m) eh ew ph pw ') #(b m) eh ew ph pw   

        equi_masks = equi_masks.unsqueeze(0).expand(b,-1,-1,-1,-1,-1)                     #b m ph pw eh ew
        equi_masks = rearrange(equi_masks, 'b m ph pw eh ew  -> (b m) ph pw eh ew ') #(b m) ph pw eh ew

        pers_coords, equi_coords = get_coords(
            pers_h, pers_w, equi_h, equi_w, cameras, pers_x.device, pers_x.dtype)
        #pers_coords: [20, 16, 16, 2]        %ori_2D [20, 16, 16, 2]  (bm h w c)
        #equi_coords: [32, 64, 2]            %ori_2D [32, 64, 2]

        #pers_x: [40, 320, 16, 8, 8]
        pers_pe = self.pe(pers_coords)      #[20, 16, 16, 320]  (m h w c)
        pers_pe = rearrange(pers_pe, 'm h w c -> m c h w')      #(m c h w) [20, 320, 16, 16]
        pers_pe = pers_pe.unsqueeze(0).expand(b,-1,-1,-1,-1)      #(b m c h w)  [2, 20, 320, 8, 8]
        pers_pe = rearrange(pers_pe, 'b m c h w -> (b m) c h w')      #(bm c h w) 
        pers_pe = pers_pe.unsqueeze(2).expand(-1,-1,f,-1,-1)      #(bm c f h w)
        pers_x_wpe = pers_x + pers_pe    #[20, 320, 24, 8, 8]

        equi_pe = self.pe(equi_coords)      #[16, 32, 320]        (h w c)
        equi_pe = repeat(equi_pe, 'h w c -> b c h w', b=b)          #[1, 320, 16, 32]
        equi_pe = equi_pe.unsqueeze(2).expand(-1,-1,f,-1,-1)      #[1, 320, 24, 16, 32]
        equi_x_wpe = equi_x + equi_pe


        # cross attention from perspective to equirectangular

        query = rearrange(equi_x, 'b c f h w -> (b f) (h w) c')
        key_value = rearrange(pers_x_wpe, '(b m) c f h w -> (b f) (m h w) c', m=m)
        

        pers_masks = repeat(pers_masks, '(b m) eh ew ph pw -> (b m) f eh ew ph pw', m=m, f=f)  #(bm f eh ew ph pw)
        pers_masks = rearrange(pers_masks, '(b m) f eh ew ph pw -> (b f) (eh ew) (m ph pw)', m=m)
        
        equi_pe = rearrange(equi_pe, 'b c f h w -> (b f) (h w) c')

        equi_x_out = self.transformer(query, key_value, mask=pers_masks, query_pe=equi_pe) ################# [24, 512, 320]

        # cross attention from equirectangular to perspective
        query = rearrange(pers_x, '(b m) c f h w -> (b f) (m h w) c', m=m, f=f)  
        key_value = rearrange(equi_x_wpe, 'b c f h w -> (b f) (h w) c', f=f)

        equi_masks = repeat(equi_masks, '(b m) ph pw eh ew -> (b m) f ph pw eh ew', m=m, f=f)  #(bm f ph pw eh ew)
        equi_masks = rearrange(equi_masks, '(b m) f ph pw eh ew -> (b f) (m ph pw) (eh ew)', m=m, f=f)

        
        pers_pe = rearrange(pers_pe, '(b m) c f h w -> (b f) (m h w) c', m=m)

        pers_x_out = self.transformer(query, key_value, mask=equi_masks, query_pe=pers_pe) #################

        
        pers_x_out = rearrange(pers_x_out, '(b f) (m h w) c -> (b m) c f h w', f=f, m=m, h=pers_h, w=pers_w) 
        equi_x_out = rearrange(equi_x_out, '(b f) (h w) c -> b c f h w', f=f, h=equi_h, w=equi_w)
    

        return pers_x_out, equi_x_out
