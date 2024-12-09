import argparse, torch, math, time, os, pickle, tqdm
import numpy as np
from datetime import datetime
from pathlib import Path

import pandas as pd

from sklearn.metrics import confusion_matrix


import torch
import torchvision.transforms as transforms

from utils.env_vars import PAIR_ABBREV
np.set_printoptions(precision=5, suppress=True)

PD_COLUMNS = ['eps', 'l2', 'linf', 'SSIM', 'Acc', 'FPR', 'FNR']
PD_COLUMNS_IMGWISE = ['path_to_img', 'eps', 'normalized_eps', 'l2', 'linf', 'SSIM', 'logit0', 'logit1', 'true_label', 'pred_label']


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test] = True
    return y_test_onehot


def random_classes_except_current(y_test, n_cls):
    y_test_new = np.zeros_like(y_test)
    for i_img in range(y_test.shape[0]):
        lst_classes = list(range(n_cls))
        lst_classes.remove(y_test[i_img])
        y_test_new[i_img] = np.random.choice(lst_classes)
    return y_test_new



class ModelSklearnLogisticRegression:
    def __init__(self, model, scaler, clip_values):
        self.scaler = scaler
        self.model = model
        self.clip_values = list(clip_values)

    def predict(self, x):
        x = np.clip(x.astype(np.float32), min(self.clip_values), max(self.clip_values))
        if len(x.shape) == 4:
            x = np.squeeze(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)
        x_preprocessed = tf_func(x)
        x_preprocessed = x_preprocessed.reshape((len(x_preprocessed), -1))
        x_preprocessed = self.scaler.transform(x_preprocessed)
        return self.model.predict_log_proba(x_preprocessed)

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits
            diff[y] = np.inf
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()


class Logger:
    def __init__(self, path, clear_file=False):
        self.path = path
        if path != '':
            folder = '/'.join(path.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)
        if clear_file == True:
            with open(self.path, 'w') as f:
                pass
        
    def print(self, message):
        print(message)
        if self.path != '':
            with open(self.path, 'a') as f:
                f.write(message + '\n')
                f.flush()


def p_selection(p_init, it, n_iters):
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def square_attack_l2(log:Logger, model:ModelSklearnLogisticRegression, x, y, corr_classified, eps, n_iters, 
                     p_init, metrics_path, targeted, loss_type):
    np.random.seed(0)

    min_val, max_val = np.min(x.flatten()), np.max(x.flatten())
    x = np.expand_dims(x, 1)
    c, h, w = x.shape[1:]
    n_features = c * h * w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    delta_init = np.zeros(x.shape)
    s = h // 5
    log.print('Initial square side={} for bumps'.format(s))
    sp_init = (h - s * 5) // 2
    center_h = sp_init + 0
    for counter in range(h // s):
        center_w = sp_init + 0
        for counter2 in range(w // s):
            delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += meta_pseudo_gaussian_pert(s).reshape(
                [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
            center_w += s
        center_h += s

    x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, min_val, max_val)

    logits = model.predict(x_best)
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(x.shape[0])

    time_start = time.time()
    s_init = int(np.sqrt(p_init * n_features / c))
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters):
        idx_to_fool = (margin_min > 0.0)
        num_to_fool = int(np.sum(idx_to_fool))

        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
        loss_min_curr = loss_min[idx_to_fool]
        delta_curr = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        s = max(int(round(np.sqrt(p * n_features / c))), 3)

        if s % 2 == 0:
            s += 1

        s2 = s + 0
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
        norms_window_2 = np.sqrt(
            np.sum(delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] ** 2, axis=(-2, -1),
                   keepdims=True))

        curr_norms_window = np.sqrt(
            np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))

        new_deltas = np.ones([x_curr.shape[0], c, s, s])
        new_deltas = new_deltas * meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
            np.maximum(eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0
        delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0

        hps_str = 's={}->{}'.format(s_init, s)
        x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * eps
        x_new = np.clip(x_new, min_val, max_val)
        curr_norms_image = np.sqrt(np.sum((x_new - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))

        logits = model.predict(x_new)
        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        mean_nq, mean_nq_ae, median_nq, median_nq_ae = np.mean(n_queries), np.mean(
            n_queries[margin_min <= 0]), np.median(n_queries), np.median(n_queries[margin_min <= 0])

        time_total = time.time() - time_start
        log.print(
            '{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} {}, n_ex={}, {:.0f}s, loss={:.3f}, max_pert={:.1f}, impr={:.0f}, num_to_fool={}'.
                format(i_iter + 1, acc, acc_corr, mean_nq_ae, median_nq_ae, hps_str, x.shape[0], time_total,
                       np.mean(margin_min), np.amax(curr_norms_image), np.sum(idx_improved), num_to_fool))
        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq, margin_min.mean(), time_total]
        if (i_iter <= 500 and i_iter % 500) or (i_iter > 100 and i_iter % 500) or i_iter + 1 == n_iters or acc == 0:
            np.save(metrics_path, metrics)
        if acc == 0:
            curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
            print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
            break

    curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
    print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))

    return n_queries, x_best


def square_attack_linf(log:Logger, model:ModelSklearnLogisticRegression, x, y, corr_classified, 
                       eps, n_iters, p_init, metrics_path, targeted, loss_type):
    np.random.seed(0)
    min_val, max_val = np.min(x.flatten()), np.max(x.flatten())
    x = np.expand_dims(x, 1)
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    x_best = np.clip(x + init_delta, min_val, max_val)

    logits = model.predict(x_best)
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')

    n_queries = np.ones(x.shape[0])

    time_start = time.time()
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters - 1):
        idx_to_fool = margin_min > 0
        num_to_fool = int(np.sum(idx_to_fool))
        x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

        x_new = np.clip(x_curr + deltas, min_val, max_val)

        logits = model.predict(x_new)
        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
        avg_margin_min = np.mean(margin_min)
        time_total = time.time() - time_start
        log.print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s), num_to_fool={}'.
            format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total, num_to_fool))

        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]
        if (i_iter <= 500 and i_iter % 20 == 0) or (i_iter > 100 and i_iter % 50 == 0) or i_iter + 1 == n_iters or acc == 0:
            np.save(metrics_path, metrics)
        if acc == 0:
            break

    return n_queries, x_best


import argparse
import pickle
from pathlib import Path
import os, json
import time

from torchvision import transforms
import numpy as np
import pandas as pd
from cytoolz import functoolz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from src.data import array_from_imgdir
from src.image_transform import ImagePerturbations, ImagePerturbationsRand
from src.image import dct, fft, log_scale

from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression

from utils.env_vars import PAIR_ABBREV
from torchmetrics import StructuralSimilarityIndexMeasure


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


def main(args):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device='cpu')
    
    square_attack = square_attack_linf if args.attack == 'square_linf' else square_attack_l2
    args.targeted = True
    args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'
    start = time.time()
    temp = REAL_DIR_NAMES.union(FAKE_DIR_NAMES)
    for img_dir_test in args.test_dirs:
        assert img_dir_test in temp
    output_dir = Path(args.output_root)
    path_to_overall_csv = os.path.join(output_dir, f"freq_domain_{args.attack}_results.csv")
    path_to_imgwise_csv = os.path.join(output_dir, f"freq_domain_{args.attack}_detailed_results.csv")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "metrics")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "adv_samples")).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(str(output_dir), 'commandline_args.txt'), 'w') as f:
        args.image_root = str(args.image_root)
        args.output_root = str(args.output_root)
        json.dump(args.__dict__, f, indent=2)
        args.image_root = Path(args.image_root)
        args.output_root = Path(args.output_root)

    log = Logger(str(os.path.join(output_dir, 'logfile.txt')), clear_file=True)
    transforms_list_base = [transforms.CenterCrop((224, 224)),
                            transforms.CenterCrop((args.crop_size, args.crop_size)),
                            transforms.ToTensor(), 
                            transforms.Grayscale()]
    transforms_list_test = transforms_list_base
        
    cache_file = Path(args.cache_file_path)

    print(f"loading already trained model, cache_file={cache_file}")
    with open(cache_file, "rb") as f:
        best_model = pickle.load(f)
    regression_model = ModelSklearnLogisticRegression(
        best_model[1], scaler=best_model[3], clip_values=(-1, 1))
    
    num_samples_per_dataset = min(int(args.num_samples_total / len(args.test_dirs)), args.num_test)
    num_samples_per_label = int(num_samples_per_dataset)
    ret_list = []
    
    for test_img_dir in args.test_dirs:
        assert cache_file.exists() == True
        ret = array_from_imgdir(
            args.image_root / test_img_dir / "val" / "val_class",
            num_samples=int(args.num_test/len(args.test_dirs)),
            num_workers=args.num_workers, 
            transforms_list=transforms_list_test
        )
        if test_img_dir in REAL_DIR_NAMES:
            ret["y_test"] = np.array([0] * len(ret["imgs"]))
        elif test_img_dir in FAKE_DIR_NAMES:
            ret["y_test"] = np.array([1] * len(ret["imgs"]))
        ret_list.append(ret)

    y_test = np.concatenate((ret_list[0]["y_test"], ret_list[1]["y_test"]))
    assert len(args.test_dirs) == 2 and 1 in set(y_test) and 0 in set(y_test)
    paths = ret_list[0]["paths"] + ret_list[1]["paths"]

    x_test = np.concatenate(
        (ret_list[0]["imgs"][0:num_samples_per_label], 
        ret_list[1]["imgs"][0:num_samples_per_label])
        )
    y_test = np.concatenate(
        (ret_list[0]["y_test"][0:num_samples_per_label], 
        ret_list[1]["y_test"][0:num_samples_per_label])
        )
    paths = ret_list[0]["paths"][0:num_samples_per_label] + ret_list[1]["paths"][0:num_samples_per_label]
    print(len(x_test))
    print(f"Finished data loading, loaded test data {','.join(args.test_dirs)}, time since start of main {time.time() - start}")
    
    label_list = np.array(y_test)
    label_numpy_inverted = 1 - label_list
    y_target = label_numpy_inverted if args.targeted else label_list
    y_target_one_hot = dense_to_onehot(y_target, 2)
    num_queries_ls, x_adv = square_attack(
            log=log, model=regression_model, x=x_test, y=y_target_one_hot, 
            corr_classified=np.array(list(range(len(label_list)))), 
            eps=args.eps, n_iters=args.n_iter, 
            p_init=args.p, metrics_path=os.path.join(output_dir, f"{args.attack}_eps_{args.eps}.npy"), 
            targeted=args.targeted, loss_type=args.loss)
    
    ssim_vals, error_l2_list, error_linf_list = [], [], []
    for bbtt in range(x_adv.shape[0]):
        ssim_vals.append(ssim(torch.tensor(x_adv[bbtt]).unsqueeze(0),
            torch.tensor(x_test[bbtt]).unsqueeze(0).unsqueeze(0)))
        error_l2_list.append(np.linalg.norm(x_adv[bbtt].flatten() - x_test[bbtt].flatten()))
        error_linf_list.append(np.max(abs(x_adv[bbtt].flatten() - x_test[bbtt].flatten())))
    
    raw_logits = regression_model.predict(x_adv)
    adv_classes = np.argmax(raw_logits, axis=1)
    file_len = 0
    if os.path.exists(path_to_imgwise_csv):
        fptr = open(path_to_imgwise_csv, "r")
        file_len = len(fptr.readlines())
        fptr.close()
    fptr = open(path_to_imgwise_csv, "a")
    if file_len == 0: 
        fptr.write(','.join(PD_COLUMNS_IMGWISE) + "\n")
    for i in range(len(paths)):
        res = ["/".join(paths[i].split("/")[-4:]), str(args.eps), str(args.eps), 
               f"{error_l2_list[i]:.6f}", f"{error_linf_list[i]:.6f}", f"{ssim_vals[i]:.6f}", 
               f"{raw_logits[i, 0]:.6f}", f"{raw_logits[i, 1]:.6f}", str(int(label_list[i])),
               str(int(adv_classes[i]))]
        fptr.write(','.join(res) + '\n')
    fptr.close()
    
    file_len = 0
    if os.path.exists(path_to_overall_csv):
        fptr = open(path_to_overall_csv, "r")
        file_len = len(fptr.readlines())
        fptr.close()
    
    fptr = open(path_to_overall_csv, "a")
    if file_len == 0: 
        fptr.write(','.join(PD_COLUMNS) + "\n")
    cmat = confusion_matrix(label_list, adv_classes, normalize="true", labels=sorted([0, 1]))
    Acc =  accuracy_score(label_list, adv_classes)
    FPR, FNR = cmat[0, 1], cmat[1, 0]
    results_str = ",".join([str(args.eps), f"{np.mean(error_l2_list):.6f}", 
                            f"{np.mean(error_linf_list):.6f}", f"{np.mean(ssim_vals):.6f}", 
                            f"{Acc:.4f}", f"{FPR:.4f}", f"{FNR:.4f}"]) + "\n"
    fptr.write(results_str)
    fptr.close()
    
    path_to_pkl = os.path.join(output_dir, "adv_samples", f"{args.attack}_eps_{args.eps}.pkl")
    if os.path.exists(path_to_pkl) == False:
        with open(path_to_pkl, 'wb') as file: 
            pickle.dump({"x_adv": x_adv, "num_queries_ls": num_queries_ls, "path_list": paths, 
                        "orig_labels": label_list, "orig_text": None}, 
                        file)
    return
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=Path, default="../fake_real_img_dataset",
                        help="Root of image directory containing 'train', 'val', and test.")
    parser.add_argument("--output_root", type=Path, required=True, help="Output directory.")
    parser.add_argument("--test_dirs", nargs="+", required=True, help="Names of directories in 'train' and 'val'.")
    parser.add_argument("--crop_size", type=int, default=64, help="Size the image will be cropped to.")
    parser.add_argument("--cache_file_path", type=str, required=True, help="Location of load model checkpoint.")
    parser.add_argument("--num_samples_total", type=int, default=100, help="Total number of samples out of args.num_test")
    parser.add_argument("--num_test", type=int, default=2000, help="max number of testing images possible")

    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers (default: 8).")
    
    parser.add_argument('--attack', type=str, required=True, choices=['square_linf', 'square_l2'], help='Attack.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                             'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.')
    parser.add_argument('--n_iter', type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    main(args)