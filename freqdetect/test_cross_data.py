import argparse, os, pickle, time
from pathlib import Path

from torchvision import transforms
import numpy as np
from cytoolz import functoolz
from sklearn.metrics import confusion_matrix, accuracy_score

from src.data import array_from_imgdir
from src.image import dct, log_scale

from utils.env_vars import PAIR_ABBREV, DATASETNAME_TO_NUMBER

REAL_DIR_NAMES = {"flickr10k", "mscoco", "googlecc_dalle3/googlecc", "textcap_gpt_11k_human"}
FAKE_DIR_NAMES = {"deepfloyd_IF_flickr30k", "stable_diffusion", "googlecc_dalle3/dalle3", "textcap_gpt_11k_synthesis_gpt4cap"}

PD_COLUMNS = ["train_ckp_use_pert", "train_sets", "train_sets_abbreviation", "test_sets", "test_sets_abbreviation", "Acc", "FNR", "FPR"]
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


def main(args):
    start = time.time()
    transforms_list_base = [transforms.CenterCrop((224, 224)),
                            transforms.CenterCrop((args.crop_size, args.crop_size)),
                            transforms.ToTensor(), 
                            transforms.Grayscale()]
    path_to_csv = os.path.join(args.output_dir, f"perturb_train_freq_domain_no_perturb_test.csv")
    
    for test_img_dir in args.test_dirs:
        real_test = PAIR_ABBREV[test_img_dir.split(",")[0]]
        fake_test = PAIR_ABBREV[test_img_dir.split(",")[1]]
        print(f"starting with {real_test}, {fake_test}")
        
        x_test, y_test = [], []
        for test_dir in test_img_dir.split(","):
            ret = array_from_imgdir(
                args.image_root / test_dir / "val" / "val_class",
                num_samples=int(1000),
                num_workers=args.num_workers, 
                transforms_list=transforms_list_base
            )
            x_test.append(ret["imgs"])

            if test_dir in REAL_DIR_NAMES:
                y_test.extend([0] * 1000)
            elif test_dir in FAKE_DIR_NAMES:
                y_test.extend([1] * 1000)
        
        x_test = np.concatenate(x_test, axis=0)
        y_test = np.array(y_test)
        print(len(x_test), len(y_test))
        print(f"Finished data loading, loaded test data {test_img_dir}, time since start of main {time.time() - start}")
        
        x_test_tf = tf_func(x_test)
        x_test_tf = x_test_tf.reshape((len(x_test_tf), -1))

        print(f"loading already trained model, cache_file={args.load_checkpoint_path}")
        with open(args.load_checkpoint_path, "rb") as f:
            best_model = pickle.load(f)
        scaler = best_model[3]
        x_test_tf = scaler.transform(x_test_tf)
        
        y_pred_test = best_model[1].predict(x_test_tf)
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
        
        Cmat = confusion_matrix(y_test, y_pred_test, normalize="true", labels=sorted([0, 1]))
        Acc = accuracy_score(y_test, y_pred_test)
        FPR, FNR = Cmat[0, 1], Cmat[1, 0]
        fptr.write(','.join(["True", train_name_comb, DATASETNAME_TO_NUMBER[train_name_comb], 
                             test_name_comb, DATASETNAME_TO_NUMBER[test_name_comb], f"{Acc:.4f}", f"{FNR:.4f}", f"{FPR:.4f}"]) 
                   + "\n")
        fptr.close()
    return
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=Path, default="../fake_real_img_dataset", help="Root of image directory containing 'train', 'val', and test.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--test_dirs", nargs="+", 
                        default=["flickr10k,deepfloyd_IF_flickr30k", "mscoco,stable_diffusion", 
                                 "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap", 
                                 "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3"], 
                        help="Test_dirs, assuming root is fake_real_img_dataset")
    parser.add_argument("--train_dir", required=True, type=str,
                        choices=["flickr10k,deepfloyd_IF_flickr30k", "mscoco,stable_diffusion", 
                                 "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap", 
                                 "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3"], 
                        help="Which train directories were used?")
    parser.add_argument("--load_checkpoint_path", type=str, required=True, help="Path to checkpoint (pickle file).")
    parser.add_argument("--crop_size", type=int, default=64, help="Size the image will be cropped to.")
    
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    real_train = PAIR_ABBREV[args.train_dir.split(",")[0]]
    fake_train = PAIR_ABBREV[args.train_dir.split(",")[1]]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
