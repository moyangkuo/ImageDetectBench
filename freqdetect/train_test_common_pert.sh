#! /bin/bash

perturb_names=("jpeg" "brightness" "contrast" "gaussian-noise" "gaussian-blur")
perturbe_names_length=${#perturb_names[@]}
for ((j=0; j<perturbe_names_length; j+=1))
do
    python train_test_common_pert.py --output_root output_dir/common_perturb/pert --cache_dir output_dir/common_perturb/cache --test_dirs flickr10k,deepfloyd_IF_flickr30k --train_dirs flickr10k,deepfloyd_IF_flickr30k --num_test 2000 --test_img_perturb "${perturb_names[j]}" --num_workers 1  # note ${array[i]}=real ${array[i+1]}=fake
    python train_test_common_pert.py --output_root output_dir/common_perturb/pert --cache_dir output_dir/common_perturb/cache --test_dirs "mscoco,stable_diffusion" --train_dirs "mscoco,stable_diffusion" --num_test 2000 --test_img_perturb "${perturb_names[j]}" --num_workers 1  # note ${array[i]}=real ${array[i+1]}=fake
    python train_test_common_pert.py --output_root output_dir/common_perturb/pert --cache_dir output_dir/common_perturb/cache --test_dirs "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap" --train_dirs "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap" --num_test 2000 --test_img_perturb "${perturb_names[j]}" --num_workers 1  # note ${array[i]}=real ${array[i+1]}=fake
    python train_test_common_pert.py --output_root output_dir/common_perturb/pert --cache_dir output_dir/common_perturb/cache --test_dirs "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3" --train_dirs "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3" --num_test 2000 --test_img_perturb "${perturb_names[j]}" --num_workers 1 # note ${array[i]}=real ${array[i+1]}=fake
done
