import os, time, json, argparse, tqdm, torch, clip
import utils.utils as utils
import numpy as np
from pathlib import Path

from sklearn.metrics import confusion_matrix

import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


from utils.image_transform import ImagePerturbationsRand
from utils.data_utils import load_dataset_pair
from utils.model_data_utils import collate_fn, NeuralNet, FC

from utils.env_vars import PERTURB_TYPE_TO_VALS

def get_args_parser():
    parser = argparse.ArgumentParser('SiT', add_help=False)
    parser.add_argument('--write_logits', default=False, type=bool, help="Write logits of torch models", action=argparse.BooleanOptionalAction)
    parser.add_argument('--device', default=0, choices=[0, 1], type=int, help="Which gpu is used.")
    parser.add_argument("--train_dirs", nargs="+", 
                        default=["flickr10k,deepfloyd_IF_flickr30k", "mscoco,stable_diffusion", 
                                 "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap", 
                                 "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3"], 
                        help="Test_dirs, assuming root is fake_real_img_dataset")
    
    parser.add_argument('--batch_size', default=250, type=int)
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')

    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate.")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target LR at the end of optimization.")

    parser.add_argument('--use_text', type=bool, default=False, help="if True, then use text.", action=argparse.BooleanOptionalAction)

    parser.add_argument('--img_size', type=int, default=224, help="size of input img")
    parser.add_argument('--load_checkpoint_path', type=str, required=True, help="From where to load in linear?")
    parser.add_argument('--output_dir', required=True, type=str, help='Path to save logs and checkpoints (and to read checkpoints when testing).')

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    return parser


def finetune(args):
    with open(os.path.join(args.output_dir, 'finetune_commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    device = torch.device("cuda:%d" % args.device if torch.cuda.is_available() else "cpu")
    perturb_type_to_vals_min_max = {k: [np.min(v), np.max(v)] for k, v in PERTURB_TYPE_TO_VALS.items()}
    transform = transforms.Compose([
        transforms.CenterCrop((args.img_size, args.img_size)),
        ImagePerturbationsRand(perturb_type_to_vals_min_max, seed=args.seed),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_datasets = []
    for train_dir in args.train_dirs:
        train_datasets.append(load_dataset_pair(transform, train=True, metadata_fname="train_fname_map_to_prompt.txt",
                                            real_name=train_dir.split(",")[0], fake_name=train_dir.split(",")[1], 
                                            root_dir="../fake_real_img_dataset"))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    print(f"Finetuning using {args.train_dirs}; total: {len(train_dataset)} data points in training set")
    model, _ = clip.load("ViT-B/32", device=device)
    if args.use_text == True:
        linear = FC(512, 2).to(device)
    else: 
        linear = NeuralNet(1024, [512, 256], 2).to(device)
    if args.load_checkpoint_path is not None and os.path.exists(args.load_checkpoint_path):
        linear.load_state_dict(torch.load(args.load_checkpoint_path, map_location=device)["linear"], strict=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, 
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256., 
        args.min_lr, args.epochs, len(train_loader), warmup_epochs=args.warmup_epochs)

    wd_schedule = utils.cosine_scheduler(args.weight_decay,
        args.weight_decay, args.epochs, len(train_loader))


    optimizer = torch.optim.Adam(linear.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    time0 = time.time()
    for epoch in range(0, args.epochs):
        train_stats = finetune_one_epoch(epoch, model, linear, train_loader, optimizer, lr_schedule, wd_schedule, criterion,
                                      device)
        
        save_dict = {'linear': linear.state_dict(), 
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch + 1, 
                     'args': args}
        
        log_stats = {**{f'finetune_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "finetune_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        print("\n\nepoch %d, running cumulative time: " % epoch, time.time() - time0, "\n")
        
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint_perturbed_imgs.pth'))
    return


def finetune_one_epoch(epoch, model, linear, train_loader, optimizer, lr_schedule, wd_schedule, criterion,
                       device):
    linear.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    all_preds = []
    all_targets = []

    for it, (data1, prompt, _, target) in tqdm.tqdm(enumerate(metric_logger.log_every(train_loader, 20, header)), total=len(train_loader)):
        it = len(train_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  
                param_group["weight_decay"] = wd_schedule[it]
        data1 = data1.to(device)
        text = clip.tokenize(list(prompt), truncate=True).to(device)
        with torch.no_grad():
            imga_embedding = model.encode_image(data1)
            text_emb = model.encode_text(text)
        img_embed, text_embed, target = imga_embedding.to(device), text_emb.to(device), target.to(device)
        if args.use_text == False:
            emb = torch.cat((img_embed, text_embed), dim=1)
            output = linear(emb.float())
        else:
            output = linear(img_embed.float())

        _, preds = torch.max(output, 1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        C = confusion_matrix(all_targets, all_preds, normalize="true")

        metric_logger.update(true_neg=C[0, 0])
        metric_logger.update(false_neg=C[1, 0])
        metric_logger.update(true_pos=C[1, 1])
        metric_logger.update(false_pos=C[0, 1])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return train_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(args)
    finetune(args)