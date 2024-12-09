import argparse, copy, tqdm, torch
import pickle
from pathlib import Path
import os, json
import datetime
import time

from torchvision import transforms
import numpy as np
import pandas as pd
from cytoolz import functoolz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torchmetrics import StructuralSimilarityIndexMeasure

from src.data import array_from_imgdir
from src.image_transform import ImagePerturbations, ImagePerturbationsRand
from src.image import dct, fft, log_scale

from utils.env_vars import PAIR_ABBREV, PERTURB_TYPE_TO_VALS, DATASETNAME_TO_NUMBER

REAL_DIR_NAMES = {"flickr10k", "mscoco", "googlecc_dalle3/googlecc", "textcap_gpt_11k_human"}
FAKE_DIR_NAMES = {"deepfloyd_IF_flickr30k", "stable_diffusion", "googlecc_dalle3/dalle3", "textcap_gpt_11k_synthesis_gpt4cap"}

INT_TO_CLASS = {1: "fake", 0: "real"}

tf_name, tf_func = "log(DCT)", functoolz.compose_left(dct, log_scale)

def get_abbreviation(train_dirs, test_dirs):
    train_str = ""
    if train_dirs is not None:
        for train_dir in train_dirs:
            train_str = train_str + PAIR_ABBREV[train_dir] + "_"
        train_str = train_str[0:-1]
    test_str = ""
    if test_dirs is not None:
        for test_dir in test_dirs:
            test_str = test_str + PAIR_ABBREV[test_dir] + "_"
        test_str = test_str[0:-1]
    return train_str, test_str


def main(args, perturb_param=None):
    start = time.time()
    temp = REAL_DIR_NAMES.union(FAKE_DIR_NAMES)
    assert len(args.train_dirs) == 1 and len(args.test_dirs) == 1
    for img_dir_train, img_dir_test in zip(args.train_dirs, args.test_dirs):
        assert img_dir_train.split(",")[0] and img_dir_train.split(",")[1] in temp
        assert img_dir_test.split(",")[0] and img_dir_test.split(",")[1] in temp
    train_abbr, test_abbr = get_abbreviation(args.train_dirs[0].split(","), args.test_dirs[0].split(","))
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cpu')

    transforms_list_base = [transforms.CenterCrop((224, 224)),
                            transforms.CenterCrop((args.crop_size, args.crop_size)),
                            transforms.ToTensor(), 
                            transforms.Grayscale()]
    transforms_list_no_perturb = transforms_list_base.copy()
    transforms_list_train, transforms_list_test = transforms_list_base.copy(), transforms_list_base.copy()
    perturb_type_to_vals_min_max = {k: [np.min(v), np.max(v)] for k, v in PERTURB_TYPE_TO_VALS.items()}

    transforms_list_train.insert(1, ImagePerturbationsRand(perturb_type_to_vals_min_max))
    transforms_list_test.insert(1, ImagePerturbations(args.test_img_perturb, perturb_param))
    
    cache_file = (
            cache_dir
            / f"{train_abbr}.pickle"
        )
    
    for test_img_dir in args.test_dirs:
        dir1, dir2 = test_img_dir.split(",")
        
        results = pd.DataFrame(columns=['pert', 'factor', 'dataset', 'dataset_name', 'Acc', 'FNR', 'FPR', 'ssim', 'TN', 'TP', 'FN', 'FP'])
        load_already_trained_model = True if (cache_file.exists() and not args.overwrite) else False

        ret_dir1 = array_from_imgdir(
            args.image_root / dir1 / "val" / "val_class", num_samples=int(args.num_test/len(args.test_dirs)), 
            num_workers=args.num_workers, transforms_list=transforms_list_test)
        ret_no_perturb_dir1 = array_from_imgdir(
            args.image_root / dir1 / "val" / "val_class", num_samples=int(args.num_test/len(args.test_dirs)),
            num_workers=args.num_workers, transforms_list=transforms_list_no_perturb)
        
        ret_dir2 = array_from_imgdir(
            args.image_root / dir2 / "val" / "val_class", num_samples=int(args.num_test/len(args.test_dirs)), 
            num_workers=args.num_workers, transforms_list=transforms_list_test)
        ret_no_perturb_dir2 = array_from_imgdir(
            args.image_root / dir2 / "val" / "val_class", num_samples=int(args.num_test/len(args.test_dirs)),
            num_workers=args.num_workers, transforms_list=transforms_list_no_perturb)
        
        x_test = np.concatenate([ret_dir1["imgs"], ret_dir2["imgs"]])
        paths_test = ret_dir1["paths"]
        paths_test.extend(ret_dir2["paths"])
        half_len_x_test = int(len(x_test)/2)
        y_test_dir1 = np.array([0] * half_len_x_test) if dir1 in REAL_DIR_NAMES else np.array([1] * half_len_x_test)
        y_test_dir2 = np.array([0] * half_len_x_test) if dir2 in REAL_DIR_NAMES else np.array([1] * half_len_x_test)
        y_test = np.concatenate([y_test_dir1, y_test_dir2])
        print(len(x_test))
        print(f"Finished data loading, loaded test data {test_img_dir}, time since start of main {time.time() - start}")

        print("Started ssim calculation...")
        x_no_perturb = np.concatenate([ret_no_perturb_dir1["imgs"], ret_no_perturb_dir2["imgs"]])        
        ssim_val = float(ssim(torch.tensor(x_test).unsqueeze(1), torch.tensor(x_no_perturb).unsqueeze(1)).cpu().item())

        print("Done with ssim calculation")

        x_test_tf = tf_func(x_test)
        x_test_tf = x_test_tf.reshape((len(x_test_tf), -1))

        if load_already_trained_model:
            print(f"loading already trained model, cache_file={cache_file}")
            with open(cache_file, "rb") as f:
                best_model = pickle.load(f)
            scaler = best_model[3]
            x_test_tf = scaler.transform(x_test_tf)
        else:
            real_trains, fake_trains = [], []
            for train_img_dir in args.train_dirs[0].split(","):
                if train_img_dir in REAL_DIR_NAMES:
                    real_trains.append(array_from_imgdir(
                        args.image_root / train_img_dir / "train" / "train_class",
                        num_samples=int(args.num_train/len(args.train_dirs)),
                        num_workers=args.num_workers, transforms_list=transforms_list_train
                    )["imgs"])
                elif train_img_dir in FAKE_DIR_NAMES:
                    fake_trains.append(array_from_imgdir(
                        args.image_root / train_img_dir / "train" / "train_class",
                        num_samples=int(args.num_train/len(args.train_dirs)),
                        num_workers=args.num_workers, transforms_list=transforms_list_train
                    )["imgs"])
            print(f"Loaded training data {args.train_dirs}")
            print(f"Time since start of main {time.time() - start}")
            real_trains, fake_trains = np.concatenate(real_trains), np.concatenate(fake_trains)
            x_train = np.concatenate([real_trains, fake_trains])
            y_train = np.array([0] * len(real_trains) + [1] * len(fake_trains))
            x_train, y_train = shuffle(x_train, y_train, random_state=0)

            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=float(args.num_val/args.num_train), random_state=0)
            del real_trains, fake_trains

            x_train_tf, x_val_tf = map(tf_func, [x_train, x_val])
            x_train_tf, x_val_tf = x_train_tf.reshape((len(x_train_tf), -1)), x_val_tf.reshape((len(x_val_tf), -1))
            scaler = StandardScaler()
            x_train_tf = scaler.fit_transform(x_train_tf)
            x_val_tf, x_test_tf = map(scaler.transform, [x_val_tf, x_test_tf])

            best_model = None
            for lmbda in [1e4, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4]:
                clf = LogisticRegression(C=1 / lmbda, max_iter=1000)
                clf.fit(x_train_tf, y_train)
                val_score = clf.score(x_val_tf, y_val)
                if best_model is None or best_model[2] < val_score:
                    best_model = (lmbda, clf, val_score, scaler)
            with open(cache_file, "wb") as f:
                pickle.dump(best_model, f)
        Acc = best_model[1].score(x_test_tf, y_test)

        y_pred_test = best_model[1].predict(x_test_tf)
        _, temp_test_abbr = get_abbreviation(None, [dir1, dir2])
        Cnorm = confusion_matrix(y_test, y_pred_test, normalize="true")
        Cvanilla = confusion_matrix(y_test, y_pred_test)
        TN, FN, TP, FP = Cvanilla[0, 0], Cvanilla[1, 0], Cvanilla[1, 1], Cvanilla[0, 1]
        FNR, FPR = Cnorm[1, 0], Cnorm[0, 1]
        pert, factor = args.test_img_perturb, perturb_param
        dataset_name = temp_test_abbr
        dataset = DATASETNAME_TO_NUMBER[dataset_name][1]

        results.loc[len(results)] = [pert, factor, dataset, dataset_name, Acc, FNR, FPR, ssim_val, TN, TP, FN, FP]
        path_to_csv = os.path.join(args.output_root, f'{dataset_name}_results.csv')
        results.to_csv(path_to_csv, mode='a', header=not os.path.exists(path_to_csv), index=False)
    return
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=Path, default="../fake_real_img_dataset",
                        help="Root of image directory containing 'train', 'val', and test.")
    parser.add_argument("--output_root", type=Path, required=True, help="Output directory.")
    parser.add_argument("--cache_dir", type=Path, required=True, help="Cached models directory.")
    parser.add_argument("--test_dirs", nargs="+", required=True, help="Names of directories in 'train' and 'val'.")
    parser.add_argument("--train_dirs", nargs="+",required=True, help="Names of directories in 'train' and 'val'.")
    parser.add_argument("--crop_size", type=int, default=64, help="Size the image will be cropped to.")
    parser.add_argument("--num_train", type=int, default=20000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument('--overwrite', default=False, type=bool, help="Recompute instead of using existing data.", 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers (default: 8).")
    parser.add_argument("--columns", nargs="+", default=["log(DCT)"], help="Columns to print.",)
    parser.add_argument('--test_img_perturb', required=True, type=str, 
                        choices=PERTURB_TYPE_TO_VALS.keys())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    l = PERTURB_TYPE_TO_VALS[args.test_img_perturb]
    print(args)
    for i in range(len(l)):
        print(args.test_img_perturb, l)
        main(args, l[i])
