import argparse, torch, math, time, os, pickle, tqdm
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.metrics import confusion_matrix, accuracy_score
from torchmetrics import StructuralSimilarityIndexMeasure

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


from utils.data_utils import load_dataset_pair
from utils.model_data_utils import CLIP_MLP_Model, collate_fn

np.set_printoptions(precision=5, suppress=True)

PD_COLUMNS = ['eps', 'l2', 'linf', 'SSIM', 'Acc', 'FPR', 'FNR']
PD_COLUMNS_IMGWISE = ['path_to_img', 'eps', 'normalized_eps', 'l2', 'linf', 'SSIM', 'logit0', 'logit1', 'true_label', 'pred_label']


def get_args_parser():
    parser = argparse.ArgumentParser('SiT', add_help=False)
    parser.add_argument('--device', default=0, help="Which gpu is used.")

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
    parser.add_argument('--load_checkpoint_path', required=True, type=str, help='Path to checkpoint.')

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')

    parser.add_argument('--attack', type=str, required=True, choices=['square_linf', 'square_l2'], help='Attack.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                             'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.')
    parser.add_argument('--n_iter', type=int, default=10000)
    return parser


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



class ModelPT:
    def __init__(self, model, batch_size, device):
        self.device = device
        self.batch_size = batch_size

        model.eval()
        self.model = model.to(self.device)

    def predict(self, x, txt):
        x = x.astype(np.float32)

        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        with torch.no_grad():
            for i in range(n_batches):
                x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
                txt_batch = txt[i*self.batch_size:(i+1)*self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch).to(self.device)
                logits = self.model((x_batch_torch, txt_batch)).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits

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


def square_attack_l2(log:Logger, model:ModelPT, x, txt, y, corr_classified, eps, n_iters, 
                     p_init, metrics_path, targeted, loss_type):
    np.random.seed(0)

    min_val, max_val = np.min(x.flatten()), np.max(x.flatten())
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

    logits = model.predict(x_best, txt)
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
        txt_curr = [txt[bababa] for bababa in np.where(idx_to_fool==True)[0]]
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

        logits = model.predict(x_new, txt_curr)
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


def square_attack_linf(log:Logger, model:ModelPT, x, txt, y, corr_classified, 
                       eps, n_iters, p_init, metrics_path, targeted, loss_type):
    np.random.seed(0)
    min_val, max_val = np.min(x.flatten()), np.max(x.flatten())
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    x_best = np.clip(x + init_delta, min_val, max_val)

    logits = model.predict(x_best, txt)
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
        txt_curr = [txt[bababa] for bababa in np.where(idx_to_fool==True)[0]]
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

        logits = model.predict(x_new, txt_curr)
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


def test(args, test_dataset_dict, checkpoint_path, save_dir):
    log = Logger(str(os.path.join(args.output_dir, 'logfile.txt')), clear_file=True)
    square_attack = square_attack_linf if args.attack == 'square_linf' else square_attack_l2
    
    path_to_overall_csv = os.path.join(args.output_dir, f"{'defake' if args.use_text else 'univ_clip_linear'}_{args.attack}_results.csv")
    path_to_imgwise_csv = os.path.join(args.output_dir, f"{'defake' if args.use_text else 'univ_clip_linear'}_{args.attack}_detailed_results.csv")
    args.targeted = True
    args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'
    transform_list = [
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    inverse_norm = transforms.Compose([
        transforms.Normalize(mean = [0., 0., 0.], std = [1/0.5, 1/0.5,51/0.5]),
        transforms.Normalize(mean = [-0.5, -0.5, -0.5], std = [1., 1., 1.])])
    transform = transforms.Compose(transform_list)
    
    if args.device != "cpu":
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device='cpu')
    model = CLIP_MLP_Model(args.use_text, device)
    model_pytorch_cls = ModelPT(
        model=model, batch_size=args.batch_size, device=device)
    
    linear_dict = torch.load(checkpoint_path, map_location=device)["linear"]
    load_msg = model.linear.load_state_dict(linear_dict, strict=True)
    print(f"loaded linear, msg={load_msg}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    assert len(args.test_dirs) == 1
    test_dir_names = args.test_dirs[0]

    img_list, text_list, path_list, label_list = [], [], [], []
    test_dataset = test_dataset_dict[test_dir_names]
    num_samples_per_dataset = min(int(args.num_samples_total / len(args.test_dirs)), len(test_dataset))
    num_samples_per_label = int(num_samples_per_dataset/2)
    for ooo in range(len(test_dataset.datasets)):
        test_dataset.datasets[ooo].transform = transform
    test_dataset_half = torch.utils.data.Subset(test_dataset, list(range(num_samples_per_label)))
    test_dataset_other_half = torch.utils.data.Subset(
        test_dataset, range(int(len(test_dataset)/2), int(len(test_dataset)/2 + num_samples_per_label)))
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_half, test_dataset_other_half])
    dl = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn,
                    num_workers=args.num_workers, pin_memory=True)
    for (img, text, path, labels) in tqdm.tqdm(dl, total=len(dl), desc="dl"):
        img_list.append(img.cpu().numpy())
        text_list.extend(text)
        path_list.extend(path)
        label_list.extend(labels.cpu().tolist())
    print("start concat")
    img_numpy = np.concatenate(img_list, axis=0)
    print("Done concatting img_numpy")
    print("label_list mean", np.mean(label_list))
    label_list = np.array(label_list)
    label_numpy_inverted = 1 - label_list
    y_target = label_numpy_inverted if args.targeted else label_list
    y_target_one_hot = dense_to_onehot(y_target, 2)
    num_queries_ls, x_adv = square_attack(
        log=log, model=model_pytorch_cls, x=img_numpy, txt=text_list, y=y_target_one_hot, 
        corr_classified=np.array(list(range(len(text_list)))), eps=args.eps, n_iters=args.n_iter, 
        p_init=args.p, metrics_path=os.path.join(args.output_dir, "metrics", f"{args.attack}_eps_{args.eps}.npy"), 
        targeted=args.targeted, loss_type=args.loss)
    ssim_vals, error_l2_list, error_linf_list = [], [], []
    
    for bbtt in range(x_adv.shape[0]):
        ssim_vals.append(float(
            ssim(inverse_norm(torch.tensor(x_adv[bbtt]).unsqueeze(0)),
            inverse_norm(torch.tensor(img_numpy[bbtt]).unsqueeze(0))).cpu().item()
        ))
        error_l2_list.append(np.linalg.norm(x_adv[bbtt].flatten() - img_numpy[bbtt].flatten()))
        error_linf_list.append(np.max(abs(x_adv[bbtt].flatten() - img_numpy[bbtt].flatten())))
    
    raw_logits = model_pytorch_cls.predict(x_adv, text_list)
    print(raw_logits.shape)
    adv_classes = np.argmax(raw_logits, axis=1)
    
    file_len = 0
    if os.path.exists(path_to_imgwise_csv):
        fptr = open(path_to_imgwise_csv, "r")
        file_len = len(fptr.readlines())
        fptr.close()
    fptr = open(path_to_imgwise_csv, "a")
    if file_len == 0: 
        fptr.write(','.join(PD_COLUMNS_IMGWISE) + "\n")
    for i in range(len(path_list)):
        res = ["/".join(path_list[i].split("/")[-4:]), str(args.eps), str(args.eps/2), 
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
    Acc = accuracy_score(label_list, adv_classes)
    FPR, FNR = cmat[0, 1], cmat[1, 0]
    results_str = ",".join([str(args.eps), f"{np.mean(error_l2_list):.6f}", 
                            f"{np.mean(error_linf_list):.6f}", f"{np.mean(ssim_vals):.6f}", 
                            f"{Acc:.4f}", f"{FPR:.4f}", f"{FNR:.4f}"]) + "\n"
    fptr.write(results_str)
    fptr.close()
    path_to_pkl = os.path.join(args.output_dir, "adv_samples", f"{args.attack}_eps_{args.eps}.pkl")
    if os.path.exists(path_to_pkl) == False:
        with open(path_to_pkl, 'wb') as file: 
            pickle.dump({"x_adv": x_adv, "num_queries_ls": num_queries_ls, "path_list": path_list, 
                        "orig_labels": label_list, "orig_text": text_list}, 
                        file)
    return


def get_test_dataset_dict(metadata_fname):
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
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "metrics")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "adv_samples")).mkdir(parents=True, exist_ok=True)
    print(args)

    test_dataset_dict = get_test_dataset_dict("val_fname_map_to_prompt.txt")
    test(args, test_dataset_dict, args.load_checkpoint_path, save_dir=args.output_dir)
