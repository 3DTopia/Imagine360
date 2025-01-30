import random
import torch
import torch.nn.functional as F
import numpy as np
import src.utils.pano_utils.Equirec2Perspec as E2P
import src.utils.pano_utils.Perspec2Equirec as P2E

import src.utils.pano_utils.multi_Perspec2Equirec as m_P2E
from src.modules.utils import get_maxrec_cord

from PIL import Image
import torchvision.transforms as transforms

def video_mask(pixel_values, pixel_pad=None):
    [_, _, _, h, w] = pixel_values.shape

    min_rect_w = int(w * 1 / 4)
    min_rect_h = int(h * 1 / 4)

    max_rect_w = int(w * 3 / 4)
    max_rect_h = int(h * 3 / 4)

    # Generate mask templete
    mask = torch.ones_like(pixel_values)[:, :, 0:1, :, :]
    mask.to(pixel_values.device)
    
    mask_choice = ['horizontal outpaint', 'vertical outpaint', 'float outpaint']

    if pixel_pad is None:
        n = random.uniform(0, 1)
        if n < 0.4:
            mask_use = ['horizontal outpaint']
        elif 0.4 <= n < 0.8:
            mask_use = ['vertical outpaint']
        else:
            mask_use = ['float outpaint']
    else:
        mask_use = 'specific'
        for idx in range(len(pixel_pad)):
            [pad_up, pad_down, pad_left, pad_right] = pixel_pad[idx].tolist()
            if pad_right == 0 and pad_left==0:
                mask[idx:(idx + 1), :, :, pad_up:-pad_down, :] = 0
            elif pad_up == 0 and pad_down==0:
                mask[idx:(idx + 1), :, :, :, pad_left:-pad_right] = 0
            else:
                mask[idx:(idx + 1), :, :, pad_up:-pad_down, pad_left:-pad_right] = 0

    if 'horizontal outpaint' in mask_use:
        rect_w = random.randint(min_rect_w, max_rect_w)
        
        rect_start_w = random.randint(0, int(w - rect_w))
        rect_end_w = int(rect_start_w + rect_w)
        
        mask[:, :, :, :, rect_start_w:rect_end_w] = 0

    elif 'vertical outpaint' in mask_use:
        rect_h = random.randint(min_rect_h, max_rect_h)

        rect_start_h = random.randint(0, int(h - rect_h))
        rect_end_h = int(rect_start_h + rect_h)

        mask[:, :, :, rect_start_h:rect_end_h, :] = 0

    elif 'float outpaint' in mask_use:
        rect_w = random.randint(min_rect_w, max_rect_w)
        
        rect_start_w = random.randint(0, int(w - rect_w))
        rect_end_w = int(rect_start_w + rect_w)

        rect_h = random.randint(min_rect_h, max_rect_h)

        rect_start_h = random.randint(0, int(h - rect_h))
        rect_end_h = int(rect_start_h + rect_h)
        
        mask[:, :, :, rect_start_h:rect_end_h, rect_start_w:rect_end_w] = 0

    return mask

def erp_mask(pixel_values, target_size, anchor_size):
    # torch.Size([1, 64, 256, 512, 3])
    # pixel_values = B,F,3,H,W
    # print(pixel_values.shape, target_size, anchor_size)
    device = pixel_values.device
    h, w = target_size
    mask = np.ones((h, w, 3)) * 255.
    mask = mask.astype(np.uint8)
    erp = E2P.Equirectangular(img_name=mask)
    fov = 90
    thetas = [0,90,180,270]
    phis = [0,-fov,fov]
    framelist = []
    F_T_P_array = []
    for th in thetas:
        for phi in phis:
            pers_img = erp.GetPerspective(fov, th, phi, anchor_size[3], anchor_size[4])
            if th == 0 and phi == 0:
                pers_img2 = np.zeros(pers_img.shape).astype(np.uint8)
                framelist.append(pers_img2)
            else:
                framelist.append(pers_img)
            F_T_P_array.append((fov,th,phi))

    pers = m_P2E.Perspective(framelist, F_T_P_array)
    erp2 = pers.GetEquirec(h,w)
    mask2 = erp2[:,:,0] > 0
    mask2 = torch.Tensor(mask2).to(device)

    points = np.where(erp2[:,:,0] == 0.0)
    points = np.array(points).T
    hmin, hmax, wmin, wmax = points[:,0].min(), points[:,0].max(), points[:,1].min(), points[:,1].max()
    anchor_h, anchor_w= hmin, wmin
    anchor_size_new  = (hmax-hmin, wmax-wmin)
    anchor_h, anchor_w, anchor_size_new = torch.Tensor([anchor_h]).to(device), torch.Tensor([anchor_w]).to(device), torch.Tensor(anchor_size_new).to(device)

    # Generate mask templete
    erp_mask = torch.ones_like(pixel_values)[:, :, 0:1, :, :]
    erp_mask[:,:,:,~(mask2.bool())] = 0
    erp_mask.to(pixel_values.device)


    return erp_mask, anchor_h, anchor_w, anchor_size_new

def pers2erp_mask(target_size, anchor_size, device):
    
    h, w = target_size
    mask = np.ones((h, w, 3)) * 255.
    erp = E2P.Equirectangular(img_name=mask)
    fov = 90
    thetas = [0,90,180,270]
    phis = [0,-fov,fov]
    framelist = []
    F_T_P_array = []
    for th in thetas:
        for phi in phis:
            pers_img = erp.GetPerspective(fov, th, phi, anchor_size, anchor_size)
            if th == 0 and phi == 0:
                pers_img2 = np.zeros(pers_img.shape)
                framelist.append(pers_img2)
            else:
                framelist.append(pers_img)
            F_T_P_array.append((fov,th,phi))

    pers = m_P2E.Perspective(framelist, F_T_P_array)
    erp2 = pers.GetEquirec(h,w)
    mask2 = erp2[:,:,0] > 0
    mask2 = torch.Tensor(mask2).to(device)
    erp2_mask = erp2 > 0

    points = np.where(erp2[:,:,0] == 0.0)
    points = np.array(points).T
    hmin, hmax, wmin, wmax = points[:,0].min(), points[:,0].max(), points[:,1].min(), points[:,1].max()
    anchor_h, anchor_w= hmin, wmin
    anchor_size_new  = (hmax-hmin, wmax-wmin)
    anchor_h, anchor_w, anchor_size_new = torch.Tensor([anchor_h]).to(device), torch.Tensor([anchor_w]).to(device), torch.Tensor(anchor_size_new).to(device)
    return mask2, anchor_h, anchor_w, anchor_size_new


def get_anchor_target(pixel_values, ph_list, fov=90, th=0):
    if len(pixel_values.shape) == 4:
        pixel_values = pixel_values.unsqueeze(0)  #[1, 64, 3, 256, 512]
    [b, f, c, h, w] = pixel_values.shape
    
    pers_size = int(h / 2)
    
    anchor_pers =[]
    
    for i in range(f):
        pixel_values_numpy = (pixel_values[0,i].permute(1,2,0).cpu().numpy() + 1)/2 * 255
        equ = E2P.Equirectangular(pixel_values_numpy.astype(np.uint8))
        input_pers = equ.GetPerspective(fov, th, ph_list[i], pers_size, pers_size)
        input_pers = input_pers.astype(np.uint8)
        anchor_pers.append(input_pers)
    
    anchor_pers = np.stack(anchor_pers)  #[f,h,w,c]
    anchor_pixels_values_pers = torch.from_numpy((anchor_pers/127.5)-1).permute(0,3,1,2).unsqueeze(0).expand(b,-1,-1,-1,-1).to(pixel_values.device)
    
    target_pixels_values = pixel_values.clone()
    
    pitchs = []
    masks = []
    anchor_pixels_values = []
    relative_positions = []
    
    for i in range(f):
    
        p2e = P2E.Perspective(input_pers, fov, th, ph_list[i])
        _, pano_mask = p2e.GetEquirec(h, w)
        
        mask = torch.from_numpy((1-pano_mask[..., :1]))
        mask = mask.permute(2,0,1).unsqueeze(0).expand(b,-1,-1,-1).float().to(pixel_values.device) #[b h w c]
        masks.append(mask)
        max_rect = get_maxrec_cord(pano_mask[...,0])
        top_left_y, top_left_x, rect_width, rect_height = max_rect


        anchor_pixels_value = pixel_values[:, i, :,  top_left_y:top_left_y + rect_height, top_left_x:top_left_x + rect_width]

        anchor_pixels_value = F.interpolate(anchor_pixels_value, size=(256, 256), mode='bilinear', align_corners=False)
        
        anchor_pixels_values.append(anchor_pixels_value)

        pitch = torch.tensor([ph_list[i]], device=pixel_values.device)   
        pitchs.append(pitch)
        
        relative_position = torch.tensor([int(h/2 - (top_left_y + top_left_y + rect_height)/2), int(w/2 - (top_left_x + top_left_x + rect_width)/2), rect_height, rect_width, h, w], device=pixel_values.device)   
        relative_positions.append(relative_position)
        
    # {Hoffset, Woffset, Hanchor, Wanchor, Htarget, Wtarget} tensor([1.0000e+00, 1.0000e+00, 2.5500e+02, 2.5500e+02, 5.1200e+02, 1.0240e+03])
    # relative_position  #tensor([[1, 1, 255, 255, 512, 1024]])S
        
    pitchs = torch.stack(pitchs, dim=1)    
    masks = torch.stack(masks, dim=1)    
    anchor_pixels_values = torch.stack(anchor_pixels_values, dim=1)
    relative_positions = torch.stack(relative_positions, dim=0)
    relative_positions = relative_positions.unsqueeze(0).repeat(b, 1, 1)
    
    return anchor_pixels_values, anchor_pixels_values_pers, target_pixels_values, masks, relative_positions, pitchs



