import torch

# Load the tensor
cka_path = "experiments/CKA/test_activations/objects_cvcl-resnext_vision_encoder.model.layer1.pt"
tensor = torch.load(cka_path, weights_only=True)

# Convert to a single tensor if needed
if isinstance(tensor, (list, tuple)):
    tensor = torch.cat([t.flatten() for t in tensor])

# Compute statistics
max_value = tensor.max().item()
min_value = tensor.min().item()
avg_value = tensor.mean().item()

print(f"CKA Values:")
print(f"Max: {max_value}")
print(f"Min: {min_value}")
print(f"Avg: {avg_value}")


# Load the tensor
clip_dissect_path = "experiments/neuron_labeling/saved_activations/objects_cvcl-resnext_vision_encoder.model.layer1.pt"
tensor = torch.load(clip_dissect_path, weights_only=True)

# Convert to a single tensor if needed
if isinstance(tensor, (list, tuple)):
    tensor = torch.cat([t.flatten() for t in tensor])

# Compute statistics
max_value = tensor.max().item()
min_value = tensor.min().item()
avg_value = tensor.mean().item()

print(f"CLIP-Dissect Activations:")
print(f"Max: {max_value}")
print(f"Min: {min_value}")
print(f"Avg: {avg_value}")