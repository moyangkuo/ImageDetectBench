import pickle, sys, tqdm
import torch
import torch.nn as nn
import torch_dct as dct
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from torchvision import transforms
import torchvision
import numpy as np
from src.image_transform import ImagePerturbationsRand
from utils.env_vars import PERTURB_TYPE_TO_VALS
from utils.data_utils import load_dataset_pair
from utils import utils 

def collate_fn(batch):
    data = [item["image"] for item in batch]
    prompt = [item["prompt"] for item in batch]
    image_path = [item["image_path"] for item in batch]
    target = [item["label"] for item in batch]
    data = torch.stack(data)
    target = torch.LongTensor(target)
    return [data, prompt, image_path, target]


class Log(nn.Module):
    def __init__(self):
        super(Log, self).__init__()
    def forward(self, x):
        return torch.log(torch.abs(x) + 1e-12)


class Normalize(nn.Module):
    def __init__(self, mu, scale):
        super(Normalize, self).__init__()
        self.mu = torch.tensor(mu)
        self.scale = torch.tensor(scale)
    def forward(self, x):
        return (x.reshape(x.shape[0], -1).squeeze() - self.mu) / self.scale


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        x = self.linear(x)
        return x

if __name__ == "__main__":
    img_size = 64
    device = torch.device("cuda:0")
    l = {"jpeg": [99, 90, 80, 60, 40, 20], 
        "brightness": [-0.5, -0.25, 0.0, 0.25, 0.5], 
        "contrast": [0.5, 0.75, 1.0, 1.25, 1.5], 
        "gaussian-noise": [0, 0.1, 0.2, 0.3, 0.4, 0.5], 
        "gaussian-blur": [0.1, 0.25, 0.5, 0.75, 1.0], }
    
    batch_size = 500
    epochs  = 20
    weight_decay = 0.01
    lr = 0.0003
    warmup_epochs = 5
    min_lr = 1e-6
    
    perturb_type_to_vals_min_max = {k: [np.min(v), np.max(v)] for k, v in PERTURB_TYPE_TO_VALS.items() if k in l.keys()}
    train_dir_names = "mscoco,stable_diffusion"
    transform = transforms.Compose([ImagePerturbationsRand(perturb_type_to_vals_min_max),
                                    transforms.ToTensor(), 
                                    transforms.CenterCrop(img_size), transforms.Grayscale(), 
                                    dct.dct_2d, Log(), 
                                    Normalize(0, 1)])

    train_dataset = load_dataset_pair(transform=transform, train=True, metadata_fname="train_fname_map_to_prompt.txt", 
                                real_name=train_dir_names.split(",")[0], fake_name=train_dir_names.split(",")[1], 
                                root_dir="../fake_real_img_dataset")
    dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn,
                                num_workers=8, pin_memory=True)
    model = LogisticRegressionModel(img_size**2).to(device)
    
    lr_schedule = utils.cosine_scheduler(
        lr * (batch_size * 1), 
        min_lr, epochs, len(dl), warmup_epochs=warmup_epochs)
    wd_schedule = utils.cosine_scheduler(weight_decay,
        weight_decay, epochs, len(dl))
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        correct = 0
        total = 0
        for it, (img, prompt, image_paths, target) in tqdm.tqdm(enumerate(dl)):
            it = len(dl) * ep + it
        
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[it]
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = crit(out, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(out, 1)
            correct += (predicted.detach().cpu() == target.cpu()).sum()
            total += len(predicted)
        print(correct / total)
        torch.save(model.state_dict(), "./output_dir/whitebox_pytorch/freq_domain_mscoco_SD_pert.pth")
