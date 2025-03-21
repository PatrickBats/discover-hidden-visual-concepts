#!/bin/bash
# filepath: run_all.sh

# Function to run a command with timing
run_command() {
    echo "============================================================"
    echo "Starting: $1"
    echo "Command: $2"
    echo "Started at: $(date)"
    echo "------------------------------------------------------------"
    
    start_time=$(date +%s)
    
    # Run the command
    $2
    status=$?
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Format duration
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    echo "------------------------------------------------------------"
    if [ $status -eq 0 ]; then
        echo "✅ Completed successfully in ${minutes}m ${seconds}s"
    else
        echo "❌ Failed with status $status after ${minutes}m ${seconds}s"
        echo "Stopping execution."
        exit 1
    fi
    echo "============================================================"
    echo ""
}

# Execute all commands in sequence
# run_command "Neuron Labeling" "./scripts/label_neuron.sh"
run_command "Baseline N-Way" "./scripts/baseline_n_way.sh"
run_command "Neuron N-Way" "./scripts/neuron_n_way.sh"
run_command "CKA for ResNext" "python -m src.compute_cka --batch_size 512 --model1 resnext"
run_command "CKA for CLIP-Res" "python -m src.compute_cka --batch_size 512 --model1 clip-res"

echo "🎉 All tasks completed successfully!"