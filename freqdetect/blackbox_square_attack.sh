#!/bin/bash

python blackbox_square_attack.py --test_dirs mscoco stable_diffusion --cache_file_path "output_dir/common_perturb/cache/best_model_train_mscoco_SD_log(DCT)_64_20000_2000.pickle" --output_root output_dir/blackbox/square/mscoco_SD --attack square_linf --num_samples_total 100 --n_iter 10000