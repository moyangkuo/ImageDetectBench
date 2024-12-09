#!/bin/bash

# Specify device in $1, or hard-code it. 

### Whitebox Attack (PGD)
# UnivCLIP
python whitebox.py --device $1 --test_dirs mscoco,stable_diffusion --output_dir path/to/whitebox_pgd --load_checkpoint_path checkpoints/path/to/univclip_checkpoint.pth --num_samples_total 100 --whitebox_attack "PGD" --pgd_iters 1000 --use_text
