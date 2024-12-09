#! /bin/bash
dataset_names=("flickr10k,deepfloyd_IF_flickr30k" "mscoco,stable_diffusion" "textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap" "googlecc_dalle3/googlecc,googlecc_dalle3/dalle3")

declare -A names_to_abbr_dict

names_to_abbr_dict["flickr10k,deepfloyd_IF_flickr30k"]="flickr_IF"
names_to_abbr_dict["mscoco,stable_diffusion"]="mscoco_SD"
names_to_abbr_dict["textcap_gpt_11k_human,textcap_gpt_11k_synthesis_gpt4cap"]="textcap_hunyuan"
names_to_abbr_dict["googlecc_dalle3/googlecc,googlecc_dalle3/dalle3"]="googlecc_dalle3"
for name in ${dataset_names[@]}
do
  python test_cross_dataset.py --batch_size 250 --device $1 --num_workers 4 --output_dir "checkpoints/defake_${names_to_abbr_dict["$name"]}" --load_checkpoint_path "checkpoints/defake_${names_to_abbr_dict["$name"]}/checkpoint.pth" --use_text
done
