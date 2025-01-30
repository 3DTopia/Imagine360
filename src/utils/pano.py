from . import py360convert
import numpy as np
from PIL import Image
import os
import cv2

import argparse
import torch.nn.functional as F
from einops import rearrange


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def random_sample_spherical(n):
    xyz = np.random.normal(size=(n, 3))
    xyz = normalize(xyz)
    return xyz


def random_sample_camera(n):
    xyz = random_sample_spherical(n)
    phi = np.arcsin(xyz[:, 2].clip(-1, 1))
    theta = np.arctan2(xyz[:, 0], xyz[:, 1])
    return theta, phi


def horizon_sample_camera(n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    phi = np.zeros_like(theta)
    return theta, phi


def icosahedron_sample_camera():
    # reference: https://en.wikipedia.org/wiki/Regular_icosahedron
    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)
    theta_step = 2.0 * np.pi / 5.0

    thetas = []
    phis = []
    for triangle_index in range(20):
        # 1) the up 5 triangles
        if 0 <= triangle_index <= 4:
            theta = - np.pi + theta_step / 2.0 + triangle_index * theta_step
            phi = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)

        # 2) the middle 10 triangles
        # 2-0) middle-up triangles
        if 5 <= triangle_index <= 9:
            triangle_index_temp = triangle_index - 5
            theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
            phi = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)

        # 2-1) the middle-down triangles
        if 10 <= triangle_index <= 14:
            triangle_index_temp = triangle_index - 10
            theta = - np.pi + triangle_index_temp * theta_step
            phi = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))

        # 3) the down 5 triangles
        if 15 <= triangle_index <= 19:
            triangle_index_temp = triangle_index - 15
            theta = - np.pi + triangle_index_temp * theta_step
            phi = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))

        thetas.append(theta)
        phis.append(phi)

    return np.array(thetas), np.array(phis)


def pad_pano(pano, padding):
    if padding <= 0:
        return pano

    if pano.ndim == 5:
        b, m = pano.shape[:2]
        pano_pad = rearrange(pano, 'b m c h w -> (b m c) h w')
    elif pano.ndim == 4:
        b = pano.shape[0]
        pano_pad = rearrange(pano, 'b c h w -> (b c) h w')
    else:
        raise NotImplementedError('pano should be 4 or 5 dim')

    pano_pad = F.pad(pano_pad, [padding, ] * 2, mode='circular')

    if pano.ndim == 5:
        pano_pad = rearrange(pano_pad, '(b m c) h w -> b m c h w', b=b, m=m)
    elif pano.ndim == 4:
        pano_pad = rearrange(pano_pad, '(b c) h w -> b c h w', b=b)

    return pano_pad


def unpad_pano(pano_pad, padding):
    if padding <= 0:
        return pano_pad
    return pano_pad[..., padding:-padding]

def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R

class Cubemap:
    def __init__(self, cubemap, cube_format):
        if cube_format == 'horizon':
            pass
        elif cube_format == 'list':
            cubemap = py360convert.cube_list2h(cubemap)
        elif cube_format == 'dict':
            cubemap = py360convert.cube_dict2h(cubemap)
        elif cube_format == 'dice':
            cubemap = py360convert.cube_dice2h(cubemap)
        else:
            raise NotImplementedError('unknown cube_format')
        assert len(cubemap.shape) == 3
        assert cubemap.shape[0] * 6 == cubemap.shape[1]
        self.cubemap = cubemap

    def to_equirectangular(self, h, w, mode='bilinear'):
        return Equirectangular(py360convert.c2e(self.cubemap, h, w, mode, cube_format='horizon'))

    @classmethod
    def from_mp3d_skybox(cls, mp3d_skybox_path, scene, view):
        keys = ['U', 'L', 'F', 'R', 'B', 'D']
        images = {}
        for idx, key in enumerate(keys):
            img_path = os.path.join(mp3d_skybox_path, scene, 'matterport_skybox_images', f"{view}_skybox{idx}_sami.jpg")
            images[key] = np.array(Image.open(img_path))
        images['R'] = np.flip(images['R'], 1)
        images['B'] = np.flip(images['B'], 1)
        images['U'] = np.flip(images['U'], 0)
        images['U'] = np.rot90(images['U'], 1)
        images['D'] = np.rot90(images['D'], 1)
        return cls(images, 'dict')


class Equirectangular:
    def __init__(self, equirectangular):
        self.equirectangular = equirectangular

    def to_cubemap(self, face_w=256, mode='bilinear'):
        return Cubemap(py360convert.e2c(self.equirectangular, face_w, mode, cube_format='horizon'), 'horizon')

    @classmethod
    def from_file(cls, img_path):
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert("RGB")
        return cls(np.array(img))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(self.equirectangular.astype(np.uint8)).save(path)

    def to_perspective(self, fov, yaw, pitch, hw, mode='bilinear'):
        return py360convert.e2p(self.equirectangular, fov, yaw, pitch, hw, mode=mode)

    def rotate(self, degree):
        if degree % 360 == 0:
            return
        self.equirectangular = np.roll(
            self.equirectangular, int(degree / 360 * self.equirectangular.shape[1]), axis=1)

    def flip(self, flip=True):
        if flip:
            self.equirectangular = np.flip(self.equirectangular, 1)

