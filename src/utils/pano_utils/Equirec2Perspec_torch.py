import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

def remap_torch(image_torch, map_x, map_y):
    """
    Differentiable remap function in PyTorch, equivalent to cv2.remap
    :param image_torch: torch.Tensor (B, C, H, W)
    :param map_x: torch.Tensor (H, W)
    :param map_y: torch.Tensor (H, W)
    :return: remapped image
    """
    # Normalize map_x and map_y to [-1, 1]
    _, _, H, W = image_torch.shape

    map_x = 2.0 * map_x / (W - 1) - 1.0
    map_y = 2.0 * map_y / (H - 1) - 1.0

    # Combine the maps and add batch and channel dimensions
    grid = torch.stack((map_x, map_y), dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # Apply grid sample
    remapped_image = F.grid_sample(
        image_torch,
        grid,
        mode="nearest", 
        # mode="bilinear", 
        # padding='border',
        padding_mode="border",
        align_corners=True,
    )  # (B, C, H, W)

    return remapped_image

class Equirectangular:
    def __init__(self, img_name, text2light=False):
        if isinstance(img_name, str):
            self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        else:
            self._img = img_name

        [_, self._height, self._width, _] = self._img.shape


    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))

        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len,width), [height,1])
        z_map = -np.tile(np.linspace(-h_len, h_len,height), [width,1]).T

        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2])
        lon = np.arctan2(xyz[:, 1] , xyz[:, 0])

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180

        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90  * equ_cy + equ_cy

        latents = self._img.permute(0,3,1,2)
        lon = torch.from_numpy(lon).to(latents.device)
        lat = torch.from_numpy(lat).to(latents.device)
        print(lon.shape, lat.shape, height, width)
        persp = remap_torch(self._img, lon, lat)
        # persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp
        

        





