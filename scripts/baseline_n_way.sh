#!/bin/bash

# change directory to the root of the project 
cd "$(dirname "$0")/.."

seeds=(0 1 2)
models=(
    'cvcl-resnext'
    'cvcl-vit'
    'cvcl-resnext-random'
    'clip'
    'clip-res'
) 
device='cuda'
num_img_per_trial=(4 8 16 32) # maxium 62, due to only have 62 classes
class_type=("seen" "full" "unseen")
csv_save_path=("./experiments/trials/results/baseline.csv")
exps_root_dir=("./experiments/trials/baseline/")
max_batch_size=800
min_batch_size=5  

for model in "${models[@]}"
do
    for class_type in "${class_type[@]}"
    do
        for num_img in "${num_img_per_trial[@]}"
        do
            # calculate dynamic batch size and round down
            batch_size=$(( 4 * max_batch_size / num_img ))
            # ensure batch size is between min_batch_size and max_batch_size
            batch_size=$(( batch_size < min_batch_size ? min_batch_size : batch_size ))
            for seed in "${seeds[@]}" 
            do 
                python -m src.trial \
                    --model "$model" \
                    --seed "$seed" \
                    --device "$device" \
                    --batch_size "$batch_size" \
                    --num_img_per_trial "$num_img" \
                    --class_type "$class_type" \
                    --csv_save_path "$csv_save_path" \
                    --exps_root_dir "$exps_root_dir" \
                    # --object_resize  # resize object            
            done
        done
    done
done