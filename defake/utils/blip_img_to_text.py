from blipmodels.blip import blip_decoder
import torch
from torchvision import transforms, datasets

from tqdm import tqdm

import os
import argparse


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_data_loaders(root, batch_size, image_size):
    data_transforms = transforms.Compose([
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    images = ImageFolderWithPaths(root, data_transforms)
    loader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=1)
    return loader


def get_blip_decoder(device):
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
        
    model = blip_decoder(pretrained=model_url, image_size=224, vit='base', 
                         med_config="path/to/blipconfig/med_config.json")
    model.eval()
    return model.to(device)
