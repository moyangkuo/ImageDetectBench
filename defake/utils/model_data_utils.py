import os, time, json, argparse, tqdm, torch, clip

import torch.nn as nn
import torch.nn.functional as F
from utils.blip_img_to_text import get_blip_decoder


def collate_fn(batch):
    data = [item["image"] for item in batch]
    prompt = [item["prompt"] for item in batch]
    image_path = [item["image_path"] for item in batch]
    target = [item["label"] for item in batch]
    data = torch.stack(data)
    target = torch.LongTensor(target)
    return [data, prompt, image_path, target]

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out
    

class FC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out
    
    
class CLIP_MLP_Model(nn.Module):
    def __init__(self, use_text:bool, device):
        super().__init__()
        self.use_text = use_text
        if use_text == False:
            self.linear = FC(512, 2).to(device)
        else: 
            self.linear = NeuralNet(1024, [512, 256], 2).to(device)
        self.clip, _ = clip.load("ViT-B/32", device=device)
        self.device = device
        self.blip = get_blip_decoder(device)
        
    def forward(self, x):
        img, prompt = x
        with torch.no_grad():
            img = img.to(self.device)
            image_embedding = self.clip.encode_image(img)
            embedding = image_embedding
            if self.use_text == True:
                torch.manual_seed(0)
                captions = self.blip.generate(img, sample=True, num_beams=3, max_length=75, min_length=25)
                text = clip.tokenize(captions, truncate=True).to(self.device)
                text_embedding = self.clip.encode_text(text)
                embedding = torch.cat((image_embedding, text_embedding), dim=1)
        output = self.linear(embedding.float())
        return output
