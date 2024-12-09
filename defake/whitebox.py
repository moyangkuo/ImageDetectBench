import os, argparse, datetime, random, tqdm, pickle, time


import numpy as np
from pathlib import Path
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import clip
from clip.model import CLIP
from torchmetrics import StructuralSimilarityIndexMeasure

from art.estimators.classification import PyTorchClassifier

from utils.data_utils import load_dataset_pair, pass_through_clip

from utils.env_vars import PAIR_ABBREV, DATASETNAME_TO_NUMBER
from utils.model_data_utils import CLIP_MLP_Model, collate_fn

PD_COLUMNS=['pert', 'rb', 'rb_normalized', 'dataset', 'dataset_name', 'Acc', 'FNR', 'FPR', 'ssim', 'linf', "l2"]
PD_COLUMNS_IMGWISE = ['path_to_img', 'eps', 'normalized_eps', 'l2', 'linf', 'SSIM', 'logit0', 'logit1', 'true_label', 'pred_label']

def get_args_parser():
    parser = argparse.ArgumentParser('SiT', add_help=False)
    parser.add_argument('--device', default=0, type=int, help="Which gpu is used.")

    parser.add_argument("--test_dirs", nargs="+", 
                        default=["flickr10k,deepfloyd_IF_flickr30k", "mscoco,stable_diffusion", 
                                 "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap", 
                                 "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3"], 
                        help="Test_dirs, assuming root is fake_real_img_dataset")
    
    parser.add_argument('--batch_size', default=100, type=int)

    parser.add_argument('--use_text', type=bool, default=False, help="univ clip settings?", action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_samples_total', type=int, default=100, help="Number of samples in total that are subject to adv attack")
    parser.add_argument('--img_size', type=int, default=224, help="size of input img")

    parser.add_argument('--output_dir', required=True, type=str, help='Dir to which files are written.')
    parser.add_argument('--load_checkpoint_path', required=True, type=str, help='Path to checkpoint')

    parser.add_argument('--whitebox_attack', required=True, choices=["PGD"],
                         type=str, help='Which whitebox attack')
    parser.add_argument('--pgd_iters', default=1000, 
                         type=int, help='How many iters to update gradient for pgd')

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    return parser


def project(param_data, backup, epsilon):
    r = param_data - backup
    r = epsilon * r

    return backup + r

def gradient_wrt_data(model, device, data, txt, lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    if type(txt) == str:
        txt = [txt]
    out = model((dat, txt))
    loss = F.cross_entropy(out.to(device),lbl.to(device))
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()

def PGD_attack(model, device, dat, txt, lbl, eps, alpha, iters, rand_start=True, clamp_min_max=(-1,1), **kwargs):
    clamp_min_max = list(clamp_min_max)
    x_nat = dat.clone().detach()

    if rand_start:
        x_nat_perturbed = x_nat + (torch.rand(x_nat.shape, device=device) * (2*eps) - eps)
        x_nat_perturbed = torch.clamp(x_nat_perturbed, min=min(clamp_min_max), max=max(clamp_min_max))
    else:
        x_nat_perturbed = torch.clone(x_nat)

    for _ in range(int(iters)):
        grad_wrt_data = gradient_wrt_data(model, device, data=x_nat_perturbed, txt=txt, lbl=lbl)
        x_nat_perturbed += torch.sign(grad_wrt_data) * alpha
        perturbation_norm = torch.norm(x_nat_perturbed - x_nat, p=float("inf"), dim=(1, 2, 3), keepdim=False)
        for idxx, n in enumerate(perturbation_norm):
            if float(n) > eps:
                c = eps / n
                x_nat_perturbed[idxx] = project(x_nat_perturbed[idxx], x_nat[idxx], c)
        x_nat_perturbed = torch.clamp(x_nat_perturbed, min=min(clamp_min_max), max=max(clamp_min_max))
    return x_nat_perturbed, lbl


def attack(args, test_dataset_dict):
    results = pd.DataFrame(columns=PD_COLUMNS)
    results_imgwise = pd.DataFrame(columns=PD_COLUMNS_IMGWISE)
    transform_list = [
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)
    rb_list = np.array([0.001, 0.002, 0.003, 0.004, 0.005])*2
    
    inverse_norm = transforms.Compose([
        transforms.Normalize(mean = [0., 0., 0.], std = [1/0.5, 1/0.5,51/0.5]),
        transforms.Normalize(mean = [-0.5, -0.5, -0.5], std = [1., 1., 1.])])
    
    if args.device != "cpu":
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model = CLIP_MLP_Model(args.use_text, device)
    
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device='cpu')
    
    linear_dict = torch.load(args.load_checkpoint_path, map_location=device)["linear"]
    load_msg = model.linear.load_state_dict(linear_dict, strict=True)
    print(f"loaded linear, msg={load_msg}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    assert len(args.test_dirs) == 1
    test_dir_names = args.test_dirs[0]
    test_dataset = test_dataset_dict[test_dir_names]
    num_samples_per_dataset = min(int(args.num_samples_total / len(args.test_dirs)), len(test_dataset))
    num_samples_per_label = int(num_samples_per_dataset/2)
    for ooo in range(len(test_dataset.datasets)):
        test_dataset.datasets[ooo].transform = transform
    test_dataset_half = torch.utils.data.Subset(test_dataset, list(range(num_samples_per_label)))
    test_dataset_other_half = torch.utils.data.Subset(
        test_dataset, range(int(len(test_dataset)/2), int(len(test_dataset)/2 + num_samples_per_label)))
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_half, test_dataset_other_half])
    
    results_path = os.path.join(args.output_dir, f"{'univ_clip_linear' if args.use_text == False else 'defake'}_{args.whitebox_attack}_results.csv")
    results_path_imgwise = os.path.join(args.output_dir, f"{'univ_clip_linear' if args.use_text == False else 'defake'}_{args.whitebox_attack}_detailed_results.csv")
    start = time.time()
    for rb in tqdm.tqdm(rb_list, total=len(rb_list)):
        dl = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, pin_memory=True)
        adv_classes, label_list, ssim_list, linf_perturb, l2_perturb = [], [], [], [], []
        path_list, logits0_list, logits1_list = [], [], []
        for img, prompt, image_paths, target in dl:
            img = img.to(device)
            path_list.extend(image_paths)
            if args.whitebox_attack == "PGD":
                x_adv = PGD_attack(model, device, dat=img, txt=prompt, lbl=target, eps=rb, 
                                alpha=1.85*rb/args.pgd_iters, iters=args.pgd_iters, 
                                rand_start=True, clamp_min_max=(-1, 1))[0]
            else:
                assert args.whitebox_attack == "PGD", f"args.whitebox_attack {args.whitebox_attack} is not PGD"
            ssim_values = [float(
                ssim(inverse_norm(x_adv[xafdg].cpu().unsqueeze(0)),
                inverse_norm(img[xafdg].cpu().unsqueeze(0))).cpu().item()
            ) for xafdg in range(x_adv.shape[0])]
            ssim_list.extend(ssim_values)
            linf = torch.norm(x_adv-img, p=float("inf"), dim=(1, 2, 3))
            linf_perturb.extend(linf.cpu().tolist())
            l2 = torch.norm(x_adv-img, p=2, dim=(1, 2, 3))
            l2_perturb.extend(l2.cpu().tolist())
            
            logits_adv = model((x_adv, prompt))
            logits0_list.extend(logits_adv[:, 0].detach().cpu().numpy())
            logits1_list.extend(logits_adv[:, 1].detach().cpu().numpy())
            pred_adv = torch.argmax(logits_adv, dim=1, keepdim=False)
            label_list.extend(target.cpu().tolist())
            adv_classes.extend(pred_adv.cpu().tolist())
        ssim_val = np.mean(ssim_list)
        adv_classes, label_list = np.array(adv_classes), np.array(label_list)
        Cnorm = confusion_matrix(label_list, adv_classes, normalize="true", labels=sorted([0, 1]))
        FNR, FPR = Cnorm[1, 0], Cnorm[0, 1]
        Acc = accuracy_score(label_list, adv_classes)
        pert = args.whitebox_attack
        linf, l2 = np.mean(linf_perturb), np.mean(l2_perturb)
        
        dataset_name = PAIR_ABBREV[test_dir_names.split(",")[0]] + "_" + PAIR_ABBREV[test_dir_names.split(",")[1]]
        dataset = DATASETNAME_TO_NUMBER[dataset_name][1]
        results.loc[len(results)] = [pert, rb, rb/2, dataset, dataset_name, Acc, FNR, FPR, ssim_val, linf, l2]
        results.to_csv(results_path, index=False)
        
        for i in range(len(path_list)):
            res = ["/".join(path_list[i].split("/")[-4:]), str(rb), str(rb/2), 
                f"{l2_perturb[i]:.6f}", f"{linf_perturb[i]:.6f}", f"{ssim_list[i]:.6f}", 
                f"{logits0_list[i]:.6f}", f"{logits1_list[i]:.6f}", str(int(label_list[i])),
                str(int(adv_classes[i]))]
            results_imgwise.loc[len(results_imgwise)] = res
        results_imgwise.to_csv(results_path_imgwise, index=False)
        print(f"Done with rb={rb}, time taken since start of program = {time.time()-start}")
    return


def get_test_dataset_dict(args, metadata_fname):
    test_dataset_dict = dict()
    for test_dir_names in args.test_dirs:
        test_dataset_dict[test_dir_names] = \
            load_dataset_pair(transform=None, train=False, metadata_fname=metadata_fname, 
                              real_name=test_dir_names.split(",")[0], fake_name=test_dir_names.split(",")[1], 
                              root_dir="../fake_real_img_dataset")
    return test_dataset_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    data_names = PAIR_ABBREV[args.test_dirs[0].split(",")[0]] + "_" + PAIR_ABBREV[args.test_dirs[0].split(",")[1]]
    if args.use_text == True:
        data_names = "defake_" + data_names
    else:
        data_names = "univ_clip_linear_" + data_names
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(args)

    test_dataset_dict = get_test_dataset_dict(args, "val_fname_map_to_prompt.txt")
    attack(args, test_dataset_dict)
