#!/bin/bash

python test_cross_data.py --output_dir "output_dir/no_perturb_test" --train_dir "flickr10k,deepfloyd_IF_flickr30k" --load_checkpoint_path "output_dir/common_perturb/cache/flickr_IF.pickle"
python test_cross_data.py --output_dir "output_dir/no_perturb_test" --train_dir "mscoco,stable_diffusion" --load_checkpoint_path "output_dir/common_perturb/cache/mscoco_SD.pickle"
python test_cross_data.py --output_dir "output_dir/no_perturb_test" --train_dir "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap" --load_checkpoint_path "output_dir/common_perturb/cache/textcap_hunyuan.pickle"
python test_cross_data.py --output_dir "output_dir/no_perturb_test" --train_dir "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3" --load_checkpoint_path "output_dir/common_perturb/cache/googlecc_dalle3.pickle"
