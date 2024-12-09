#!/bin/bash

python blackbox_hop_skip_jump.py --test_dirs mscoco stable_diffusion --cache_file_path "output_dir/common_perturb/cache/best_model_train_mscoco_SD_log(DCT)_64_20000_2000.pickle" --output_root output_dir/blackbox/hop_skip_jump/mscoco_SD --norm inf --num_samples_total 100 --use_init_img
