#!/bin/bash

# change directory to the root of the project
cd "$(dirname "$0")/.."

# Models to run net-dissect on
models=(
    'cvcl-resnext'
    'resnext'
)

# Device to use
device='cuda:4'

for model in "${models[@]}"
do
    echo "Running Net-Dissect for model: $model"
    
    # Modify settings.py to set the current model
    sed -i "s/MODEL = '.*'/MODEL = '$model'/" net-dissect/settings.py
    
    # Run net-dissect
    cd net-dissect
    python main.py
    cd ..
    
    # Optional: copy or organize results
    mkdir -p experiments/net-dissect_results/
    # cp -r experiments/net-dissect_results/${model}_* experiments/net-dissect_results/organized/$model/
done