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

class Perspective:
    def __init__(self, img_name , FOV, THETA, PHI ):
        if isinstance(img_name, str):
            self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        else:
            self._img = img_name
            

        [_, self._height, self._width, _] = self._img.shape # B, H, W, 3

        
        self.wFOV = FOV   # usually set horizon to fov(90 here)
        self.THETA = THETA
        self.PHI = PHI
        self.hFOV = float(self._height) / self._width * FOV

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

    

    def GetEquirec(self,height,width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        x,y = np.meshgrid(np.linspace(-180, 180, width),np.linspace(90, -90, height))
        
        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map,y_map,z_map),axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height , width, 3])
        inverse_mask = np.where(xyz[:,:,0]>0,1,0)

        xyz[:,:] = xyz[:,:]/np.repeat(xyz[:,:,0][:, :, np.newaxis], 3, axis=2)
        
        
        lon_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(xyz[:,:,1]+self.w_len)/2/self.w_len*self._width,0)
        lat_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height,0)
        mask = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),1,0)

        # persp = cv2.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        latents = self._img.permute(0,3,1,2)
        lon = torch.from_numpy(lon).to(latents.device)
        lat = torch.from_numpy(lat).to(latents.device)
        persp = remap_torch(self._img, lon, lat)

        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)            
        persp = persp * mask
        
        
        return persp , mask
        






