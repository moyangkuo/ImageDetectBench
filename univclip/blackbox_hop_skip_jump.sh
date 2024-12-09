#!/bin/bash

python blackbox_hop_skip_jump.py --device $1 --test_dirs "mscoco,stable_diffusion"  --load_checkpoint_path checkpoints/path/to/univclip_checkpoint.pth --output_dir checkpoints/path_to_out_dir_univclip_hsj --num_samples_total 100
