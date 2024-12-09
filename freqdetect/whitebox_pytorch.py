import torch
import torch.nn as nn
import torch.nn.functional as F


def project(param_data, backup, epsilon):
    r = param_data - backup
    r = epsilon * r
    return backup + r


# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model, device, data, lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    
    out = model(Normalize(0, 1)(Log()(dct.dct_2d(dat))))
    # print(torch.mean(torch.abs(out.detach().clone()), dim=1, keepdim=True))
    out /= (2*torch.mean(torch.abs(out.detach().clone()), dim=1, keepdim=True))
    loss = F.cross_entropy(out.to(device), lbl.to(device))
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(model, device, dat, txt, lbl, eps, alpha, iters, rand_start=True, clamp_min_max=(-1,1), **kwargs):
    clamp_min_max = list(clamp_min_max)
    x_nat = dat.clone().detach()
    x_nat_std = torch.std(x_nat, dim=(-1, -2))
    if rand_start:
        x_nat_perturbed = x_nat + (torch.rand(x_nat.shape, device=device) * (2*eps) - eps)
        x_nat_perturbed = torch.clamp(x_nat_perturbed, min=min(clamp_min_max), max=max(clamp_min_max))
    else:
        x_nat_perturbed = torch.clone(x_nat)
    grad_wrt_data_norm_list = []
    for _ in range(int(iters)):
        grad_wrt_data = gradient_wrt_data(model, device, data=x_nat_perturbed, lbl=lbl)
        grad_wrt_data_norm_list.append(torch.norm(grad_wrt_data.detach().cpu(), p=2, dim=(-1, -2), keepdim=False))
        x_nat_perturbed += torch.sign(grad_wrt_data) * alpha * 1e0
        # x_nat_perturbed += grad_wrt_data * alpha
        perturbation_norm = torch.norm(x_nat_perturbed - x_nat, p=float("inf"), dim=(1, 2,), keepdim=False)
        for idxx, n in enumerate(perturbation_norm):
            if float(n) > eps:
                c = eps / n
                x_nat_perturbed[idxx] = project(x_nat_perturbed[idxx], x_nat[idxx], c)
        x_nat_perturbed = torch.clamp(x_nat_perturbed, min=min(clamp_min_max), max=max(clamp_min_max))
    grad_norm = torch.vstack(grad_wrt_data_norm_list)
    x_nat_perturbed_norm = torch.norm(x_nat_perturbed, p=2, dim=(-1, -2), keepdim=False)
    # print(grad_norm.shape)
    grad_norm = torch.mean(grad_norm, dim=0)
    # print(grad_norm.shape)
    return x_nat_perturbed, grad_norm, x_nat_perturbed_norm, x_nat_std


import pickle, sys, tqdm, argparse
import torch
import torch.nn as nn
import torch_dct as dct
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from pathlib import Path

from torchvision import transforms
import torchvision
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure
from env_vars import DATASETNAME_TO_NUMBER
sys.path.append("../defake")
from data_utils import load_dataset_pair

PD_COLUMNS=['pert', 'rb', 'rb_normalized', 'dataset', 'dataset_name', 'Acc', 'FNR', 'FPR', 'ssim', 'linf', "l2"]
PD_COLUMNS_IMGWISE = ['path_to_img', 'eps', 'normalized_eps', 'l2', 'linf', 'SSIM', 'logit0', 'logit1', 'logit0_no_pert', 'logit1_no_pert', 'true_label', 'pred_label', 'grad_norm', 'tf_img_norm', 'x_nat_std']


def collate_fn(batch):
    # item: a dict with keys "image", "prompt", "label"
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
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        x = self.linear(x)
        return x

if __name__ == "__main__":
    results_path_imgwise = "./output_dir/whitebox_pytorch/freq_domain_PGD_detailed_results.csv"
    results_path = "./output_dir/whitebox_pytorch/freq_domain_PGD_results.csv"

    num_samples_total = 100
    num_samples_per_label = int(num_samples_total/2)
    device = 0
    device = torch.device(f'cuda:{device}')
    pgd_iters = 500
    dataset_name = "mscoco_SD"
    pert = "PGD"

    # Initialize the PyTorch model
    model = LogisticRegressionModel(4096)
    model.load_state_dict(
        torch.load("./output_dir/whitebox_pytorch/freq_domain_mscoco_SD_pert.pth", weights_only=True))

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(64), transforms.Grayscale()])
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device='cpu')

    test_dir_names = "mscoco,stable_diffusion"
    test_dataset = load_dataset_pair(transform=None, train=False, metadata_fname="val_fname_map_to_prompt.txt", 
                                real_name=test_dir_names.split(",")[0], fake_name=test_dir_names.split(",")[1], 
                                root_dir="../fake_real_img_dataset")
    for ooo in range(len(test_dataset.datasets)):
        test_dataset.datasets[ooo].transform = transform

    test_dataset_half = torch.utils.data.Subset(test_dataset, list(range(num_samples_per_label)))
    test_dataset_other_half = torch.utils.data.Subset(
        test_dataset, range(int(len(test_dataset)/2), int(len(test_dataset)/2 + num_samples_per_label)))
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_half, test_dataset_other_half])
    results = pd.DataFrame(columns=PD_COLUMNS)
    results_imgwise = pd.DataFrame(columns=PD_COLUMNS_IMGWISE)
        
    logits_normal0_list, logits_normal1_list = [], []
    rb_list = [0.001, 0.002, 0.003, 0.004, 0.005]
    for idx, rb in enumerate(rb_list):
        dl = DataLoader(test_dataset, shuffle=False, batch_size=100, collate_fn=collate_fn,
                        num_workers=1, pin_memory=True)
        adv_classes, label_list, ssim_list, linf_perturb, l2_perturb = [], [], [], [], []
        path_list, logits0_list, logits1_list = [], [], []
        x_adv_list , grad_norm_list, x_nat_perturbed_norm_list = [], [], []
        x_nat_std_list = []
        for idx, (img, prompt, image_paths, target) in enumerate(dl):
            # img = img.to(device).squeeze(0)
            img = img.to(device).squeeze()
            if idx == 0:
                with torch.no_grad():
                    logits_normal = model(Normalize(0, 1)(Log()(dct.dct_2d(img))))
                    logits_normal0_list.extend(logits_normal[:, 0].detach().cpu().numpy())
                    logits_normal1_list.extend(logits_normal[:, 1].detach().cpu().numpy())
                    pred_adv = torch.argmax(logits_normal, dim=1, keepdim=False).detach().cpu().numpy()
                    print("No attack, ", confusion_matrix(target, pred_adv, normalize='true'))
                    print("no Attack: ", accuracy_score(target, pred_adv))
            path_list.extend(image_paths)
            x_adv, grad_norm, x_nat_perturbed_norm, x_nat_std = PGD_attack(model, device, dat=img, txt=prompt, lbl=target, eps=rb, 
                                        alpha=5*rb/pgd_iters, iters=pgd_iters, 
                                        rand_start=False, clamp_min_max=(0, 1))
            x_nat_std_list.extend(x_nat_std.detach().cpu().numpy())
            x_adv_list.append(x_adv)
            grad_norm = grad_norm.detach().cpu().numpy()
            grad_norm_list.extend(grad_norm)
            x_nat_perturbed_norm_list.extend(x_nat_perturbed_norm.detach().cpu().numpy())
            logits_adv = model(Normalize(0, 1)(Log()(dct.dct_2d(x_adv))))
            pred_adv = torch.argmax(logits_adv, dim=1, keepdim=False)
            print(idx, accuracy_score(target, pred_adv.detach().cpu().numpy()))
            ssim_values = [float(
                    ssim(x_adv[xafdg].cpu().unsqueeze(0).unsqueeze(0),   # range should be 1 thanks to inverse norm
                    img[xafdg].cpu().unsqueeze(0).unsqueeze(0)).cpu().item()
            ) for xafdg in range(x_adv.shape[0])]
            ssim_list.extend(ssim_values)
            linf = torch.norm(x_adv-img, p=float("inf"), dim=(1, 2))
            linf_perturb.extend(linf.cpu().tolist())
            l2 = torch.norm(x_adv-img, p=2, dim=(1, 2))
            l2_perturb.extend(l2.cpu().tolist())
            
            logits0_list.extend(logits_adv[:, 0].detach().cpu().numpy())
            logits1_list.extend(logits_adv[:, 1].detach().cpu().numpy())
            pred_adv = torch.argmax(logits_adv, dim=1, keepdim=False)
            label_list.extend(target.cpu().tolist())
            adv_classes.extend(pred_adv.cpu().tolist())

        x_adv_list = torch.cat(x_adv_list, dim=0)
        logits_adv = model(Normalize(0, 1)(Log()(dct.dct_2d(x_adv_list))))
        pred_adv = torch.argmax(logits_adv, dim=1, keepdim=False).detach().cpu()
        ssim_val = np.mean(ssim_list)
        adv_classes, label_list = np.array(pred_adv), np.array(label_list)
        print(adv_classes)
        Cnorm = confusion_matrix(label_list, adv_classes, normalize="true", labels=sorted([0, 1]))
        FNR, FPR = Cnorm[1, 0], Cnorm[0, 1]
        Acc = accuracy_score(label_list, adv_classes)
        linf, l2 = np.mean(linf_perturb), np.mean(l2_perturb)
        
        dataset = DATASETNAME_TO_NUMBER[dataset_name][1]  # [1] since the number d in front (i.e.,) "d1"
        results.loc[len(results)] = [pert, rb, rb, dataset, dataset_name, Acc, FNR, FPR, ssim_val, linf, l2]
        results.to_csv(results_path, index=False)
        
        for i in range(len(path_list)):
            res = ["/".join(path_list[i].split("/")[-4:]), str(rb), str(rb), 
                f"{l2_perturb[i]:.6f}", f"{linf_perturb[i]:.6f}", f"{ssim_list[i]:.6f}", 
                f"{logits0_list[i]:.6f}", f"{logits1_list[i]:.6f}", 
                f"{logits_normal0_list[i]:.6f}", f"{logits_normal1_list[i]:.6f}", 
                str(int(label_list[i])),
                str(int(adv_classes[i])), 
                f"{grad_norm_list[i]:.8f}", f"{x_nat_perturbed_norm_list[i]:.4f}", 
                f"{x_nat_std_list[i]:.4f}"]
            results_imgwise.loc[len(results_imgwise)] = res
        results_imgwise.to_csv(results_path_imgwise, index=False)