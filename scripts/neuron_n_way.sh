#!/bin/bash
seeds=(0 1 2)
models=("cvcl-resnext") # "clip-res" "cvcl-resnext" 
device="cuda:0"
num_img_per_trial=(4 8 16) # 62 is set due to only have 62 classes
class_type=("seen" "unseen")

# neuron concepts suffix
suffix="konk_baby+konk+30k"
csv_save_path="supply_material.csv"

# model and map_file prefix
declare -A model_map_prefixes=(
    ["clip-res"]="clip_res"
    ["cvcl-resnext-random"]="cvcl"
    ["cvcl-resnext"]="cvcl"
    ["resnext"]="resnext"
    ["dino_say_resnext50"]="dino_say"
    ["dino_s_resnext50"]="dino_s"
    ["dino_a_resnext50"]="dino_a"
    ["dino_y_resnext50"]="dino_y"
)

for class_type in "${class_type[@]}"
do
    for model in "${models[@]}"
    do
        # generate map_file according to the selected model and suffix
        map_file="./experiments/neuron_labeling/labeled_neurons/${model_map_prefixes[$model]}_${suffix}.csv"

        for num_img in "${num_img_per_trial[@]}"
        do
            # calculate dynamic batch size and round down
            batch_size=$(( (4 * 128 + num_img - 1) / num_img ))
            # ensure batch size is at least 1
            batch_size=$((batch_size > 0 ? batch_size : 1))
            for seed in "${seeds[@]}" 
            do 
                python -m src.trial --model $model --top_k 1 --map_file $map_file --seed $seed --device $device --batch_size $batch_size --num_img_per_trial $num_img --class_type $class_type --csv_save_path $csv_save_path # --object_resize #resize object
            done
        done
    done
done