#! /bin/bash

python train.py --num_workers 4 --output_dir checkpoints/common_perturb/defake_textcap_hunyuan_finetune --load_checkpoint_path path/to/checkpoint.pth --train_dirs "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap"
python train.py --num_workers 4 --output_dir checkpoints/common_perturb/defake_mscoco_SD_finetune --load_checkpoint_path path/to/checkpoint.pth --train_dirs "mscoco,stable_diffusion"
python train.py --num_workers 4 --output_dir checkpoints/common_perturb/defake_googlecc_dalle3_finetune --load_checkpoint_path path/to/checkpoint.pth --train_dirs "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3"
python train.py --num_workers 4 --output_dir checkpoints/common_perturb/defake_flickr_IF_finetune --load_checkpoint_path path/to/checkpoint.pth --train_dirs "flickr10k,deepfloyd_IF_flickr30k"
