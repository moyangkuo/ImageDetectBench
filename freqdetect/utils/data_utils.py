import os, tqdm, torch
from PIL import Image
from typing import Tuple, List

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from natsort import index_natsorted

import pandas as pd 
import clip
REAL_DIR_NAMES = {"flickr10k", "mscoco", "googlecc_dalle3/googlecc", "textcap_gpt_11k_human"}
FAKE_DIR_NAMES = {"deepfloyd_IF_flickr30k", "stable_diffusion", "googlecc_dalle3/dalle3", "textcap_gpt_11k_synthesis_gpt4cap"}


class real_or_fake(Dataset):
    def __init__(self, root_dir1, label, train : bool, transform, metadata_fname):
        if train == True:
            self.root_dir1 = os.path.join(root_dir1, "train/train_class")
            captions_file = os.path.join(root_dir1, "train/%s" % metadata_fname)
        else:
            self.root_dir1 = os.path.join(root_dir1, "val/val_class")
            captions_file = os.path.join(root_dir1, "val/%s" % metadata_fname)
        self.transform = transform
        
        with open(captions_file, 'r') as file:
            lines = file.readlines()
        self.captions = [','.join(caption.split(",")[2:])[1:-2].strip() for caption in lines]
        self.image_filenames1 = [caption.split(",")[1] for caption in lines]
        self.sorted_idx = index_natsorted(self.image_filenames1)

        self.image_filenames1 = [self.image_filenames1[i] for i in self.sorted_idx]
        self.captions = [self.captions[i] for i in self.sorted_idx]

        self.label = label

    def __len__(self):
        return len(self.image_filenames1)

    def __getitem__(self, idx):
        class_name1 = self.image_filenames1[idx]
        image_path1 = os.path.join(self.root_dir1, class_name1)
        image1 = Image.open(image_path1).convert("RGB")
        
        image1 = self.transform(image1)
        
        prompt = self.captions[idx] if idx < len(self.captions) else ""
        if len(prompt) > 3 and prompt[-3:] == "...":
            prompt = prompt[0:-3]

        return {"image": image1, "prompt": prompt, "image_path": image_path1, "label": self.label}


def load_dataset_pair(transform:transforms, train:bool, metadata_fname:str, real_name:str, fake_name:str, 
                      root_dir):
    assert real_name in REAL_DIR_NAMES and fake_name in FAKE_DIR_NAMES
    dataset_fake = real_or_fake(root_dir1=os.path.join(root_dir, fake_name), 
                                label=1, train=train, transform=transform, metadata_fname=metadata_fname)
    dataset_real = real_or_fake(root_dir1=os.path.join(root_dir, real_name), 
                                label=0, train=train, transform=transform, metadata_fname=metadata_fname)
    return torch.utils.data.ConcatDataset([dataset_fake, dataset_real])



class TensorDatasetWithPaths(Dataset[Tuple[Tensor, ...]]):
    tensors: Tuple[Tensor, ...]
    def __init__(self, *tensors: Tuple[Tensor, List]) -> None:
        for tensor in tensors:
            assert len(tensors[0]) == len(tensor) , f"Size mismatch between tensors len(tensors[0])={len(tensors[0])}, len(tensor)={len(tensor)}"
        self.tensors = tensors
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    def __len__(self):
        return len(self.tensors[0])

def pass_through_clip(data_loader, model, device):
    img_embeddings, text_embeddings, image_paths, targets = [], [], [], []
    for data1, prompt, image_path, target in tqdm.tqdm(data_loader):
        data1 = data1.to(device)
        text = clip.tokenize(list(prompt), truncate=True).to(device)
        with torch.no_grad():
            imga_embedding = model.encode_image(data1)
            text_emb = model.encode_text(text)
        img_embeddings.append(imga_embedding)
        text_embeddings.append(text_emb)
        targets.append(target)
        image_paths.extend(image_path)
    img_embeddings = torch.cat(img_embeddings, dim=0).to('cpu')
    text_embeddings = torch.cat(text_embeddings, dim=0).to('cpu')
    targets = torch.cat(targets, dim=0).to('cpu')
    return TensorDatasetWithPaths(img_embeddings, text_embeddings, image_paths, targets)

