import os, argparse, tqdm
import numpy as np
from pathlib import Path
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score
from utils.blip_img_to_text import get_blip_decoder


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import clip

from utils.data_utils import load_dataset_pair

from utils.env_vars import PAIR_ABBREV, DATASETNAME_TO_NUMBER
from utils.model_data_utils import collate_fn, CLIP_MLP_Model


PD_COLUMNS = ["train_ckp_use_pert", "train_sets", "train_sets_abbreviation", "test_sets", "test_sets_abbreviation", "Acc", "FNR", "FPR"]

def get_args_parser():
    parser = argparse.ArgumentParser('test_CLIP_vanilla_data', add_help=False)
    parser.add_argument('--device', default=0, type=int, help="Which gpu is used.")

    parser.add_argument("--test_dirs", nargs="+", 
                        default=["flickr10k,deepfloyd_IF_flickr30k", "mscoco,stable_diffusion", 
                                 "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap", 
                                 "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3"], 
                        help="Test_dirs, assuming root is fake_real_img_dataset")
    parser.add_argument("--train_dir", required=True, type=str,
                        choices=["flickr10k,deepfloyd_IF_flickr30k", "mscoco,stable_diffusion", 
                                 "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap", 
                                 "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3"], 
                        help="Test_dirs, assuming root is fake_real_img_dataset")
    
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--use_text', type=bool, default=False, help="whether to use text or not", action=argparse.BooleanOptionalAction)
    parser.add_argument('--img_size', type=int, default=224, help="size of input img")
    parser.add_argument('--output_dir', required=True, type=str, help='Dir to which files are written.')
    parser.add_argument('--load_checkpoint_path', required=True, type=str, help='Path to checkpoint')
    
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    return parser

def test(args, test_dataset_dict, checkpoint_path):
    real_train = PAIR_ABBREV[args.train_dir.split(",")[0]]
    fake_train = PAIR_ABBREV[args.train_dir.split(",")[1]]
    
    transform_list = [
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)
    device = torch.device("cuda:%d" % args.device if torch.cuda.is_available() else "cpu")
    
    model = CLIP_MLP_Model(args.use_text, device)
    linear_dict = torch.load(checkpoint_path, map_location=device)["linear"]
    load_msg = model.linear.load_state_dict(linear_dict, strict=True)
    print(load_msg)
    
    path_to_csv = os.path.join(args.output_dir, f"perturb_train_{'defake' if args.use_text == True else 'univ_clip_linear'}_no_perturb_test.csv")
    
    for test_dir_names in args.test_dirs:
        real_test = PAIR_ABBREV[test_dir_names.split(",")[0]]
        fake_test = PAIR_ABBREV[test_dir_names.split(",")[1]]
        print(f"starting with {real_test}, {fake_test}")
        test_dataset = test_dataset_dict[test_dir_names]
        for ooo in range(len(test_dataset.datasets)):
            test_dataset.datasets[ooo].transform = transform
        
        dl = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, pin_memory=True)
        Cmat, Acc = test_helper(model, dataloader=dl)
        file_len = 0
        if os.path.exists(path_to_csv):
            fptr = open(path_to_csv, "r")
            file_len = len(fptr.readlines())
            fptr.close()
        
        fptr = open(path_to_csv, "a")
        if file_len == 0: 
            fptr.write(','.join(PD_COLUMNS) + "\n")
        test_name_comb = f"{real_test}_{fake_test}"
        train_name_comb = f"{real_train}_{fake_train}"
        FPR, FNR = Cmat[0, 1], Cmat[1, 0]
        fptr.write(','.join(["True", train_name_comb, DATASETNAME_TO_NUMBER[train_name_comb], 
                             test_name_comb, DATASETNAME_TO_NUMBER[test_name_comb], f"{Acc:.4f}", f"{FNR:.4f}", f"{FPR:.4f}"]) 
                   + "\n")
        fptr.close()
    return


def test_helper(model:CLIP_MLP_Model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    image_paths_list = []
    outputs_list = []
    with torch.no_grad():
        for data1, _, image_paths, targets in tqdm.tqdm(dataloader, total=len(dataloader)):
            outputs = model((data1, None))
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            image_paths_list.extend(image_paths)
            outputs_list.append(outputs)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    C = confusion_matrix(all_targets, all_preds, normalize="true")
    Acc = accuracy_score(all_targets, all_preds)
    return C, Acc


def get_test_dataset_dict(metadata_fname):
    test_dataset_dict = dict()
    for test_dir_names in args.test_dirs:
        test_dataset_dict[test_dir_names] = \
            load_dataset_pair(transform=None, train=False, metadata_fname=metadata_fname,
                                        real_name=test_dir_names.split(",")[0], 
                                        fake_name=test_dir_names.split(",")[1], 
                                        root_dir="../fake_real_img_dataset")
    return test_dataset_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(args)
    real_train = PAIR_ABBREV[args.train_dir.split(",")[0]]
    fake_train = PAIR_ABBREV[args.train_dir.split(",")[1]]
    test_dataset_dict = get_test_dataset_dict("val_fname_map_to_prompt.txt")
    test(args, test_dataset_dict, args.load_checkpoint_path)
