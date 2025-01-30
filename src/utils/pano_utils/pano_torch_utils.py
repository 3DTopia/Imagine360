"""
- Based on code from https://github.com/fuenwang/Equirec2Perspec
- Modified to work with torch tensors (by Yuseung Lee)    
"""
import math

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


def xyz_to_lonlat_torch(xyz):
    """
    Converts XYZ coordinates to longitude-latitude using PyTorch
    :param xyz: A tensor of shape (..., 3) representing XYZ coordinates
    :return: A tensor of shape (..., 2) representing longitude and latitude
    """
    # Replace np.arctan2 and np.arcsin with torch.atan2 and torch.asin
    atan2 = torch.atan2
    asin = torch.asin

    # Replace np.linalg.norm with torch.linalg.norm
    norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)

    # Replace np.concatenate with torch.cat
    out = torch.cat([lon, lat], dim=-1)
    return out

def lonlat_to_xyz_torch(lonlat):
    """
    Converts longitude-latitude to XYZ coordinates using PyTorch
    :param lonlat: A tensor of shape (..., 2) representing longitude and latitude
    :return: A tensor of shape (..., 3) representing XYZ coordinates on the unit sphere
    """
    # Replace np.sin and np.cos with torch.sin and torch.cos
    sin = torch.sin
    cos = torch.cos

    # Replace np.linalg.norm with torch.linalg.norm
    lon = lonlat[..., 0:1]
    lat = lonlat[..., 1:]

    x = cos(lat) * sin(lon)
    y = sin(lat)
    z = cos(lat) * cos(lon)

    # Replace np.concatenate with torch.cat
    out = torch.cat([x, y, z], dim=-1)

    # fix the range of lon and lat
    for _ in range(10):
        lon[lon > torch.pi] -= 2 * torch.pi
        lon[lon < -torch.pi] += 2 * torch.pi
        lat[lat > torch.pi / 2] -= torch.pi
        lat[lat < -torch.pi / 2] += torch.pi
    return out

def project_xyz_torch(xyz, K, eps=1e-6):
    """
    Projects XYZ coordinates to z=1 plane using PyTorch
    :param xyz: A tensor of shape (..., 3) representing XYZ coordinates
    :param K: A tensor of shape (3, 3) representing the camera intrinsic matrix
    :return: A tensor of shape (..., 2) representing projected coordinates
    """
    out = xyz.clone()

    # valid point is where z > eps
    valid_mask = out[..., 2] > eps

    # Nanfy
    out[~valid_mask] = float("nan")
    out[valid_mask] /= out[valid_mask][..., 2:3]

    assert torch.allclose(out[valid_mask][..., 2:3], torch.ones_like(out[valid_mask][..., 2:3])), "z should be 1"

    # project only valid coordinates
    out[valid_mask] = torch.matmul(out[valid_mask], K.T)

    return out[..., :2]

def lonlat_to_xy_torch(lonlat, shape):
    """
    Converts longitude-latitude to pixel coordinates (X, Y) using PyTorch
    :param lonlat: A tensor of shape (..., 2) representing longitude and latitude
    :param shape: A tuple or list representing the shape of the target image (height, width)
    :return: A tensor of shape (..., 2) representing pixel coordinates
    """
    # Converting longitude and latitude to pixel coordinates
    X = (lonlat[..., 0:1] / (2 * torch.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / torch.pi + 0.5) * (shape[0] - 1)

    # Concatenate X and Y to get the output
    out = torch.cat([X, Y], dim=-1)

    return out

def xy_to_lonlat_torch(xy, shape):
    """
    Converts pixel coordinates (X, Y) to longitude-latitude using PyTorch
    :param xy: A tensor of shape (..., 2) representing x and y coordinates
    :param shape: A tuple or list representing the shape of the source image (height, width)
    :return: A tensor of shape (..., 2) representing longitude and latitude
    """
    # Converting pixel coordinates to longitude and latitude
    lon = (xy[..., 0:1] / (shape[1] - 1) - 0.5) * 2 * torch.pi
    lat = (xy[..., 1:] / (shape[0] - 1) - 0.5) * torch.pi

    # Concatenate lon and lat to get the output
    out = torch.cat([lon, lat], dim=-1)

    return out


def compute_xy_map_per_angle(
    FOV, THETA, PHI, pers_height, pers_width, panorama_height, panorama_width, device=None
):
    # Camera intrinsic matrix calculation
    f = 0.5 * pers_width / math.tan(0.5 * FOV * math.pi / 180)
    cx = (pers_width - 1) / 2.0
    cy = (pers_height - 1) / 2.0

    K = torch.tensor(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    K_inv = torch.inverse(K)

    # Generate grid of coordinates
    x = torch.arange(pers_width).repeat(pers_height, 1).float()
    y = torch.arange(pers_height).unsqueeze(1).repeat(1, pers_width).float()
    z = torch.ones_like(x).float()

    xyz = torch.stack([x, y, z], dim=-1)
    xyz = torch.matmul(xyz, K_inv.T)

    # Rotation matrices
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    R1, _ = cv2.Rodrigues(y_axis.numpy() * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = torch.tensor(R2 @ R1, dtype=torch.float32)
    xyz = torch.matmul(xyz, R.T)

    # Convert to lonlat coordinates and then to image coordinates
    lonlat = xyz_to_lonlat_torch(xyz)  # You will need to implement this in PyTorch
    XY = lonlat_to_xy_torch(
        lonlat, shape=(panorama_height, panorama_width)
    )  # You will need to implement this in PyTorch
    if device is not None:
        XY = XY.to(device)
    return XY

def compute_inverse_xy_map(
    FOV, THETA, PHI, pers_height, pers_width, panorama_height, panorama_width, device=None
):
    # Camera intrinsic matrix calculation
    f = 0.5 * pers_width / math.tan(0.5 * FOV * math.pi / 180)
    cx = (pers_width - 1) / 2.0
    cy = (pers_height - 1) / 2.0

    K = torch.tensor(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    # Generate grid of coordinates
    x = torch.arange(panorama_width).repeat(panorama_height, 1).float()
    y = torch.arange(panorama_height).unsqueeze(1).repeat(1, panorama_width).float()

    lonlat = xy_to_lonlat_torch(
        torch.stack([x, y], dim=-1), shape=(panorama_height, panorama_width)
    )

    # Rotation lonlat with PHI and THETA
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    R1, _ = cv2.Rodrigues(y_axis.numpy() * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = torch.tensor(R2 @ R1, dtype=torch.float32)
    xyz = lonlat_to_xyz_torch(lonlat)
    xyz = torch.matmul(xyz, R)

    # Convert to image coordinates
    XY = project_xyz_torch(xyz, K)

    if device is not None:
        XY = XY.to(device)
    return XY

def compute_mask_torch(
    FOV, THETA, PHI, pers_height, pers_width, panorama_height, panorama_width, device=None
):
    """
    Compute mask for the inverse mapping
    :return: mask of shape (panorama_height, panorama_width)
    """
    inv_xy_map = compute_inverse_xy_map(FOV, THETA, PHI, pers_height, pers_width, panorama_height, panorama_width, device)
    mask = (inv_xy_map[:,:,0]>0) & (inv_xy_map[:,:,0]<pers_width) & (inv_xy_map[:,:,1]>0) & (inv_xy_map[:,:,1]<pers_height)
    return mask


"""
(z_t,i) -> x_t^i
"""
def pano_to_perspective_torch(
    panorama, FOV, THETA, PHI, pers_height=512, pers_width=512
):
    """
    Panorama to perspective projection in PyTorch
        - FOV: degree
        - THETA: z-axis angle (right direction is positive, left direction is negative)
        - PHI: y-axis angle (up direction positive, down direction negative)
        - pers_height, pers_width: output image dimensions
    """
    pano_height, pano_width = panorama.shape[-2], panorama.shape[-1]
    XY = compute_xy_map(
        FOV,
        THETA,
        PHI,
        pers_height,
        pers_width,
        panorama_height=pano_height,
        panorama_width=pano_width,
    )
    XY = XY.to(panorama.device)

    persp = remap_torch(panorama, XY[..., 0], XY[..., 1])
    return persp, XY


# def convert_to_parallel_perspective(
"""
(x_t^i, i) -> z_t^i
"""
def perspective_to_pano_torch(xy_map, perspective_image, ori_image, i_idx, j_idx):
    _, _, HO, WO = ori_image.shape
    _, _, H, W = perspective_image.shape

    x = torch.clamp(xy_map[0, i_idx, j_idx].round(), 0, WO - 1).long()
    y = torch.clamp(xy_map[1, i_idx, j_idx].round(), 0, HO - 1).long()

    ori_image[:, :, y, x] = perspective_image[:, :, i_idx, j_idx].to(ori_image)

    return ori_image

@torch.no_grad()
def wrapper_perspective_to_pano_torch(perspective_image, xy_map, panorama_height, panorama_width):
    """
    Input:
        perspective_image: [1,c,pers_height,pers_width]
        xy_map: [pers_height, pers_width, 2]
    Output:
        panorama: [1,c,pano_height,pano_width]
        mask: [1,1,pano_height,pano_width]
    """
    B, C, pers_height, pers_width = perspective_image.shape
    panorama = torch.zeros(B, C, panorama_height, panorama_width).to(perspective_image)
    
    i_idx, j_idx = torch.meshgrid(
        torch.arange(pers_height), torch.arange(pers_width)
    )
    i_idx, j_idx = list(map(lambda x:x.to(perspective_image.device), [i_idx, j_idx]))

    x = torch.clamp(xy_map[i_idx, j_idx, 0].round(), 0, panorama_width - 1).long()
    y = torch.clamp(xy_map[i_idx, j_idx, 1].round(), 0, panorama_height - 1).long()

    panorama[:, :, y, x] = perspective_image[:, :, i_idx, j_idx]
    
    mask = torch.zeros(B, 1, panorama_height, panorama_width).to(panorama.device)
    mask[:, :, y, x] = 1
    return panorama, mask

    