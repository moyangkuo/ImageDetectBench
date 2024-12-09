import os, copy, tqdm, random, time, argparse, clip
import numpy as np
from pathlib import Path
import pandas as pd

from utils.blip_img_to_text import get_blip_decoder

from sklearn.metrics import confusion_matrix, accuracy_score
from torchmetrics import StructuralSimilarityIndexMeasure

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from utils.image_transform import ImagePerturbations
from utils.data_utils import load_dataset_pair, pass_through_clip

from utils.env_vars import PAIR_ABBREV, PERTURB_TYPE_TO_VALS, DATASETNAME_TO_NUMBER
from utils.model_data_utils import collate_fn, CLIP_MLP_Model


def get_args_parser():
    parser = argparse.ArgumentParser('SiT', add_help=False)
    parser.add_argument('--device', default=0, type=int, help="Which gpu is used.")

    parser.add_argument("--test_dirs", nargs="+", 
                        default=["flickr10k,deepfloyd_IF_flickr30k", "mscoco,stable_diffusion", 
                                 "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap", 
                                 "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3"], 
                        help="Test_dirs, assuming root is fake_real_img_dataset")
    
    parser.add_argument('--test_img_perturb', default=['jpeg', 'brightness', 'contrast', 'gaussian-noise', 'gaussian-blur'], nargs="+")
    parser.add_argument('--batch_size', default=500, type=int)

    parser.add_argument('--use_text', type=bool, default=False, help="use text or not?", action=argparse.BooleanOptionalAction)

    parser.add_argument('--img_size', type=int, default=224, help="size of input img")

    parser.add_argument('--output_dir_main_results', required=True, type=str, help='Dir to which the pandas df is written.')
    parser.add_argument('--load_checkpoint_path', required=True, type=str, help='Path to checkpoint.')

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    return parser


def test(args, test_img_perturb, test_dataset_dict, checkpoint_path, 
         test_img_perturb_param=None):
    transforms_no_normalize_no_perturb = transforms.Compose([
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor()])
    transforms_no_normalize_with_perturb = transforms.Compose([
        transforms.CenterCrop((args.img_size, args.img_size)),
        ImagePerturbations(test_img_perturb, test_img_perturb_param if test_img_perturb != "jpeg" else int(test_img_perturb_param)),
        transforms.ToTensor()])
    transforms_with_normalize_with_perturb = transforms.Compose([
        transforms.CenterCrop((args.img_size, args.img_size)),
        ImagePerturbations(test_img_perturb, test_img_perturb_param if test_img_perturb != "jpeg" else int(test_img_perturb_param)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    device = torch.device("cuda:%d" % args.device if torch.cuda.is_available() else "cpu")
    model = CLIP_MLP_Model(use_text=args.use_text, device=device)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)

    linear_dict = torch.load(checkpoint_path, map_location=device)["linear"]
    print(model.linear.load_state_dict(linear_dict, strict=True))
    print("loaded linear")


    for test_dir_names in args.test_dirs:
        print(test_dir_names, test_img_perturb, test_img_perturb_param)
        results = pd.DataFrame(columns=['pert', 'factor', 'dataset', 'dataset_name', 'Acc', 'FNR', 'FPR', 'ssim', 'TN', 'TP', 'FN', 'FP'])
        test_dataset = test_dataset_dict[test_dir_names]
        test_dataset_no_perturb = copy.deepcopy(test_dataset_dict[test_dir_names])
        for ooo in range(len(test_dataset.datasets)):
            test_dataset.datasets[ooo].transform = transforms_no_normalize_with_perturb

        for ooo in range(len(test_dataset_no_perturb.datasets)):
            test_dataset_no_perturb.datasets[ooo].transform = transforms_no_normalize_no_perturb
        ssim_chosen_idx = list(range(len(test_dataset)))
        random.seed(0)
        random.shuffle(ssim_chosen_idx)
        ssim_chosen_idx = ssim_chosen_idx[0:200]
        ssim_val = float(ssim(torch.stack([test_dataset[cx]["image"].to(device) for cx in ssim_chosen_idx]), 
                                torch.stack([test_dataset_no_perturb[cx]["image"].to(device) for cx in ssim_chosen_idx])).cpu().item())

        for ooo in range(len(test_dataset.datasets)):
            test_dataset.datasets[ooo].transform = transforms_with_normalize_with_perturb
        dl = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, pin_memory=True)
        
        ret_dict = test_helper(model, dataloader=dl)
        all_targets, all_preds = ret_dict["all_targets"], ret_dict["all_preds"]
        Cnorm = confusion_matrix(all_targets, all_preds, normalize="true")
        Cvanilla = confusion_matrix(all_targets, all_preds)
        Acc = accuracy_score(all_targets, all_preds)
        TN, FN, TP, FP = Cvanilla[0, 0], Cvanilla[1, 0], Cvanilla[1, 1], Cvanilla[0, 1]
        FNR, FPR = Cnorm[1, 0], Cnorm[0, 1]
        pert, factor = test_img_perturb, test_img_perturb_param
        dataset_name = PAIR_ABBREV[test_dir_names.split(",")[0]] + "_" + PAIR_ABBREV[test_dir_names.split(",")[1]]
        dataset = DATASETNAME_TO_NUMBER[dataset_name][1]

        results.loc[len(results)] = [pert, factor, dataset, dataset_name, Acc, FNR, FPR, ssim_val, TN, TP, FN, FP]
        path_to_csv = os.path.join(args.output_dir_main_results, f'{dataset_name}_results.csv')
        results.to_csv(path_to_csv, mode='a', header=not os.path.exists(path_to_csv), index=False)
    return


def test_helper(model, dataloader):
    all_preds = []
    all_targets = []
    outputs_list = []
    with torch.no_grad():
        for img, txt, _, targets in tqdm.tqdm(dataloader, total=len(dataloader)):
            
            
            outputs = model((img, txt))
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            outputs_list.append(outputs)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return {"all_targets": all_targets, "all_preds": all_preds}


def get_test_dataset_dict(metadata_fname, args):
    test_dataset_dict = dict()
    for test_dir_names in args.test_dirs:
        test_dataset_dict[test_dir_names] = \
            load_dataset_pair(transform=None, train=False, metadata_fname=metadata_fname,
                                        real_name=test_dir_names.split(",")[0], 
                                        fake_name=test_dir_names.split(",")[1], 
                                        root_dir="../fake_real_img_dataset")
    return test_dataset_dict


def test_time_perturb_driver(args):
    metadata_fname = "val_fname_map_to_prompt_blip.txt"
    test_dataset_dict = get_test_dataset_dict(metadata_fname, args)
    start = time.time()
    print("testing with these perturbations: ", args.test_img_perturb)
    for perturb_name in args.test_img_perturb:
        assert perturb_name in PERTURB_TYPE_TO_VALS.keys(), f"{perturb_name} not in PERTURB_TYPE_TO_VALS.keys():"
    for perturb_name in args.test_img_perturb:
        l = PERTURB_TYPE_TO_VALS[perturb_name]
        for param in l:
            test(args, test_img_perturb=perturb_name, test_dataset_dict=test_dataset_dict, 
                    test_img_perturb_param=param, checkpoint_path=args.load_checkpoint_path)
            print(f"Done with {perturb_name} at {param} for {args.test_dirs}; time={time.time()-start}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir_main_results).mkdir(parents=True, exist_ok=True)
    print(args)
    test_time_perturb_driver(args)
