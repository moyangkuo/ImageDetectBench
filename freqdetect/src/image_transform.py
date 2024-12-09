from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random, time
import os
import io
from skimage.filters import gaussian
import skimage as sk
import cv2
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import warnings
from functools import partial
from uuid import uuid4, uuid5
warnings.filterwarnings('ignore')

import torch 
import torch.nn as nn
from typing import Tuple
from torch.nn.functional import conv2d
import torchvision.transforms as transforms
import torchvision.transforms.functional
import pillow_heif


FROST_DIR = './'


def to_float32(image):
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def to_uint8(image):
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


class GaussianNoise(nn.Module):
    # input must be tensor, scaling agnostic
    def __init__(self, std, device=None):
        super(GaussianNoise, self).__init__()
        self.noise_level = std

    def forward(self, watermarked_image):
        self.min_value = torch.min(watermarked_image)
        self.max_value = torch.max(watermarked_image)
        ### Add gaussian noise
        gaussian = torch.randn_like(watermarked_image)
        noised_image = watermarked_image + self.noise_level * gaussian
        noised_image = noised_image.clamp(self.min_value.item(), self.max_value.item())
        return noised_image


def jpeg_compression(x, c):
    output = io.BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)
    return x


class Brightness(nn.Module):
    def __init__(self, brightness):
        super(Brightness, self).__init__()
        self.contrast = 1
        self.brightness = brightness

    def forward(self, watermarked_img):
        self.min_value = torch.min(watermarked_img)
        self.max_value = torch.max(watermarked_img)
        noised_img = torch.clamp((self.contrast * watermarked_img + self.brightness), self.min_value.item(), self.max_value.item())
        return noised_img


class Contrast(nn.Module):
    def __init__(self, contrast=1.0):
        super(Contrast, self).__init__()
        self.contrast = contrast

    def forward(self, x):
        means = torch.mean(x, dim=(0, 1), keepdim=True)
        return torch.clamp((x - means) * self.contrast + means, 0, 1)


def gaussian_blur(x, c):
    return np.clip(cv2.GaussianBlur(src=np.array(x), ksize=(129, 129), sigmaX=float(c), sigmaY=float(c)), 0, 255)


def defocus_blur(x, c):
    x = np.array(x) / 255.
    kernel = disk(radius=c, alias_blur=0.5)

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def fog(x, c):
    x = np.array(x) / 255.
    orig_shape = np.max(x.shape)
    x = cv2.resize(x, (256, 256))
    max_val = x.max()
    x += c * plasma_fractal(mapsize=x.shape[0], wibbledecay=2.0)[:x.shape[0], :x.shape[0]][..., np.newaxis]
    return cv2.resize(np.clip(x * max_val / (max_val + c), 0, 1), (orig_shape, orig_shape)) * 255


def frost(x, c):
    idx = np.random.randint(5)
    filename = [os.path.join(FROST_DIR, 'frost/frost1.png'),
                os.path.join(FROST_DIR, 'frost/frost2.png'),
                os.path.join(FROST_DIR, 'frost/frost3.png'),
                os.path.join(FROST_DIR, 'frost/frost4.jpg'),
                os.path.join(FROST_DIR, 'frost/frost5.jpg'),
                os.path.join(FROST_DIR, 'frost/frost6.jpg')][idx]
    frost_img = cv2.imread(filename)
    frost_img = cv2.resize(frost_img, (int(1.5 * np.array(x).shape[0]), int(1.5 * np.array(x).shape[1])))
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost_img.shape[0] - np.array(x).shape[0]), np.random.randint(0, frost_img.shape[1] - np.array(x).shape[1])
    frost_img = frost_img[x_start:x_start + np.array(x).shape[0], y_start:y_start + np.array(x).shape[1]][..., [2, 1, 0]]

    return np.clip(np.array(x) + c * frost_img, 0, 255)


def glass_blur(x, c):
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c, channel_axis=2) * 255)
    # locally shuffle pixels
    for i in range(2):
        for h in range(x.shape[0] - 1, 1, -1):
            for w in range(x.shape[1] - 1, 1, -1):
                dx, dy = np.random.randint(-1, 1, size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c, channel_axis=2), 0, 1) * 255


class RayleighNoise(nn.Module):
    def __init__(self, sigma):
        super(RayleighNoise, self).__init__()
        self.sigma = sigma
        self.np_rng = np.random.default_rng()

    def forward(self, x:torch.Tensor):
        self.min_value = torch.min(x).item()
        self.max_value = torch.max(x).item()
        rayleigh_noise = self.np_rng.rayleigh(scale=self.sigma, size=x.numpy().shape)# - 0*self.sigma * np.sqrt(np.pi / 2)
        return torch.clamp(x + rayleigh_noise, self.min_value, self.max_value)


def hue(x, c):
    return torchvision.transforms.functional.adjust_hue(x, c)


def saturation(x, c):
    return torchvision.transforms.functional.adjust_saturation(x, c)


def heif_compression(x, c):
    fname = f"/dev/shm/{uuid5(uuid4(), 'b')}.heif"  # USING THIS because io.BytesIO does not work!!!!!!
    y = pillow_heif.from_pillow(x)
    y.save(fname, quality=c)
    heif_file = pillow_heif.read_heif(fname)
    if os.path.exists(fname):
        os.remove(fname)
    y = np.asarray(heif_file)
    return Image.fromarray(y)  # 0 is lowest, 100 is highest


class ImagePerturbations:
    def __init__(self, img_transform_name:str, param):
        super().__init__()
        self.img_transform_name = img_transform_name
        
        self.param = int(param) if img_transform_name == 'jpeg' or img_transform_name == 'heif' else float(param)
        if img_transform_name == 'jpeg':
            self.func = jpeg_compression
        elif img_transform_name == 'brightness':
            self.func = Brightness(self.param)
        elif img_transform_name == 'contrast':
            self.func = Contrast(self.param)
        elif img_transform_name == 'gaussian-noise':
            self.func = GaussianNoise(abs(self.param) if self.param != 0 else 1e-9)
        elif img_transform_name == 'gaussian-blur':
            self.func = gaussian_blur
        # elif img_transform_name == 'glass-blur':
        #     self.func = glass_blur
        # elif img_transform_name == 'defocus-blur':
        #     self.func = defocus_blur
        elif img_transform_name == 'elastic-blur':
            self.func = transforms.ElasticTransform(alpha=abs(self.param))
        elif img_transform_name == 'heif':
            self.func = heif_compression
        elif img_transform_name == 'rayleigh-noise':
            self.func = RayleighNoise(sigma=abs(self.param))
        # elif img_transform_name == 'hue':
        #     self.func = hue
        # elif img_transform_name == 'saturation':
        #     self.func = saturation
        # elif img_transform_name == 'fog-blur':
        #     self.func = fog
        # elif img_transform_name == 'frost-blur':
        #     self.func = frost
        return

    def __call__(self, img):
        if self.img_transform_name == 'jpeg' or self.img_transform_name == 'heif':
            self.param = int(self.param)
        if self.img_transform_name == 'gaussian-noise':
            return transforms.ToPILImage()(self.func(transforms.ToTensor()(img)))
        elif self.img_transform_name == 'contrast':
            return transforms.ToPILImage()(self.func(transforms.ToTensor()(img)))
        elif self.img_transform_name == 'brightness':
            return transforms.ToPILImage()(self.func(transforms.ToTensor()(img)))
        elif self.img_transform_name == 'elastic-blur':
            return self.func(img)
        elif self.img_transform_name == 'rayleigh-noise':
            return transforms.ToPILImage()(self.func(transforms.ToTensor()(img)))
        return Image.fromarray(np.uint8(self.func(img, self.param)))


class ImagePerturbationsRand:
    def __init__(self, perturb_type_to_vals_min_max, seed=0):
        super().__init__()
        self.perturb_types = list(perturb_type_to_vals_min_max.keys())
        self.perturb_types.append(None)
        self.perturb_type_to_vals_min_max = perturb_type_to_vals_min_max
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, img):
        img_transform_name = self.rng.choice(self.perturb_types)
        if img_transform_name is None:  # identity transformation
            return img
        param_val = self.rng.uniform(np.min(self.perturb_type_to_vals_min_max[img_transform_name]), 
                                     np.max(self.perturb_type_to_vals_min_max[img_transform_name]))
        return ImagePerturbations(img_transform_name, param_val)(img)
