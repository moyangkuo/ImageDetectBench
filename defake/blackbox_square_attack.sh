#!/bin/bash

python blackbox_square_attack.py --device $1 --test_dirs "mscoco,stable_diffusion" --use_text --output_dir checkpoints/path/to/defake_checkpoint.pth --load_checkpoint_path checkpoints/path_to_out_dir_defake_square --num_samples_total 100 --attack square_linf --n_iter 10000 --use_text
