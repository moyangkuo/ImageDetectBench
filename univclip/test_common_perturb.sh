#! /bin/bash

python test_common_perturb.py --batch_size 50 --device $1 --num_workers 0 --output_dir_main_results checkpoints/common_perturb/univ_clip_linear --load_checkpoint_path checkpoints/common_perturb/univ_clip_linear_textcap_hunyuan_finetune/checkpoint_perturbed_imgs.pth --test_dirs textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap
python test_common_perturb.py --batch_size 50 --device $1 --num_workers 0 --output_dir_main_results checkpoints/common_perturb/univ_clip_linear --load_checkpoint_path checkpoints/common_perturb/univ_clip_linear_flickr_IF_finetune/checkpoint_perturbed_imgs.pth --test_dirs flickr10k,deepfloyd_IF_flickr30k
python test_common_perturb.py --batch_size 50 --device $1 --num_workers 0 --output_dir_main_results checkpoints/common_perturb/univ_clip_linear --load_checkpoint_path checkpoints/common_perturb/univ_clip_linear_googlecc_dalle3_finetune/checkpoint_perturbed_imgs.pth --test_dirs googlecc_dalle3/googlecc,googlecc_dalle3/dalle3
python test_common_perturb.py --batch_size 50 --device $1 --num_workers 0 --output_dir_main_results checkpoints/common_perturb/univ_clip_linear --load_checkpoint_path checkpoints/common_perturb/univ_clip_linear_mscoco_SD_finetune/checkpoint_perturbed_imgs.pth --test_dirs mscoco,stable_diffusion

