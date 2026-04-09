import os
import argparse
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# # Import your models

from safetensors.torch import load_model

import sys

# Add the directory containing your repositories
sys.path.append("/home/yhchoi")

# Now you can use a regular import
from BioFoundation.models.LUNA import LUNA
from LILIE.models.models import LILIE  # Adjust import path if needed based on your directory structure

class EndToEndLongitudinal(torch.nn.Module):
    """
    Wraps LUNA and LILIE together to allow gradients to flow 
    from the final classification output all the way back to the raw EEG input.
    """
    def __init__(self, luna_model, lilie_model):
        super().__init__()
        self.luna = luna_model
        self.lilie = lilie_model
        
        # Ensure models are in eval mode to disable dropout/batchnorm
        self.luna.eval()
        self.lilie.eval()

    def forward(self, eeg_0, eeg_1, loc_emb_0, loc_emb_1):
        # Extract embeddings using LUNA
        embed_0, _ = self.luna(eeg_0, None, loc_emb_0)
        embed_1, _ = self.luna(eeg_1, None, loc_emb_1)
        
        # Pool and classify using LILIE
        pooled = self.lilie.pooler(embed_0, embed_1)
        logits = self.lilie.clf(pooled)
        return logits

def load_raw_eeg_snippet(hdf5_path, group_name, start_idx, window_size=7680):
    """Loads a specific raw EEG snippet from the HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        data = f[group_name]['eeg'][:, start_idx : start_idx + window_size]
        data = data.astype(np.float32)
    return torch.from_numpy(data).unsqueeze(0) # Add batch dimension: (1, Channels, Time)

def compute_eeg_saliency(model, eeg_0, eeg_1, loc_emb, target_class=1):
    """
    Computes the smoothed input gradients for both longitudinal EEG inputs.
    """
    # Enable gradient tracking on raw inputs
    eeg_0.requires_grad = True
    eeg_1.requires_grad = True

    # Expand location embeddings for batch size of 1
    loc_emb_batch = loc_emb.unsqueeze(0)

    # Forward pass through the stitched model
    logits = model(eeg_0, eeg_1, loc_emb_batch, loc_emb_batch)
    
    # Get the prediction for the target class
    target_score = logits[0, target_class]
    
    # Backward pass to calculate gradients
    model.zero_grad()
    target_score.backward()

    # Extract gradients, take absolute value to get magnitude of influence
    grad_0 = eeg_0.grad.cpu().numpy()[0]
    grad_1 = eeg_1.grad.cpu().numpy()[0]

    # Apply temporal smoothing to create a CAM-like contiguous heatmap
    smoothed_cam_0 = gaussian_filter1d(np.abs(grad_0), sigma=10, axis=1)
    smoothed_cam_1 = gaussian_filter1d(np.abs(grad_1), sigma=10, axis=1)

    return logits.detach().cpu().numpy(), smoothed_cam_0, smoothed_cam_1

def plot_eeg_with_cam(eeg_data, cam_data, channel_names, title, save_path):
    """Plots the EEG channels with the CAM heatmap overlaid in the background."""
    fig, ax = plt.subplots(figsize=(16, 10))
    n_channels, n_time = eeg_data.shape
    time_axis = np.arange(n_time)

    # Normalize CAM globally for the specific plot so colors are relative across channels
    cam_min, cam_max = np.min(cam_data), np.max(cam_data)
    cam_norm = (cam_data - cam_min) / (cam_max - cam_min + 1e-8)

    offset = 0
    y_ticks = []
    
    for i in range(n_channels):
        # Normalize signal for visualization purposes
        signal = eeg_data[i, :]
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # Plot the EEG line
        ax.plot(time_axis, signal + offset, color='black', linewidth=0.8)
        
        # Overlay the heatmap as a background strip for this channel
        extent = [0, n_time, offset - 2.5, offset + 2.5]
        ax.imshow(
            cam_norm[i, :].reshape(1, -1), 
            cmap='jet', 
            aspect='auto', 
            extent=extent, 
            alpha=0.5 # Transparency so the signal remains visible
        )
        
        y_ticks.append(offset)
        offset += 6 # Vertical spacing between channels

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(channel_names)
    ax.set_xlabel("Timepoints")
    ax.set_title(title)
    
    # Add colorbar reference
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Importance (Gradient Magnitude)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM / Saliency Maps for LILIE Model")
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to HDF5 data')
    parser.add_argument('--luna_safetensor', type=str, required=True, help='Path to pretrained LUNA safetensors')
    parser.add_argument('--lilie_ckpt', type=str, required=True, help='Path to trained LILIE lightning checkpoint (.ckpt)')
    parser.add_argument('--location_emb_path', type=str, required=True, help='Path to channel location embeddings')
    parser.add_argument('--target_class', type=int, default=1, help='Class index to calculate gradients for (0 or 1)')
    
    # Parameters for the specific sample you want to visualize
    parser.add_argument('--group_0', type=str, required=True, help='HDF5 Group name for Timepoint 0: The "Before"')
    parser.add_argument('--start_idx_0', type=int, default=0, help='Start index for Timepoint 0: The "Before"')
    parser.add_argument('--group_1', type=str, required=True, help='HDF5 Group name for Timepoint 1: The "After"')
    parser.add_argument('--start_idx_1', type=int, default=0, help='Start index for Timepoint 1: The "After"')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Setup Models
    print("Loading LUNA model...")
    luna_specs = {
        'patch_size': 40, 'num_queries': 4, 'embed_dim': 64, 'depth': 8,
        'num_heads': 2, 'mlp_ratio': 4., 'norm_layer': torch.nn.LayerNorm,
        'drop_path': 0.1, 'num_classes': 1
    }
    luna_model = LUNA(**luna_specs)
    luna_model.classifier = torch.nn.Identity()
    load_model(luna_model, args.luna_safetensor, strict=False)

    print("Loading LILIE model...")
    lilie_model = LILIE.load_from_checkpoint(args.lilie_ckpt, 
                                             input_dim=256, # Adjust based on LUNA variant
                                             embedding_size=256, 
                                             num_classes=2, 
                                             pool_method="Attentive", 
                                             clf_method="NN")
    
    combined_model = EndToEndLongitudinal(luna_model, lilie_model).to(device)

    # 2. Load Data & Embeddings
    location_embeddings = torch.load(args.location_emb_path).to(device)
    
    eeg_0 = load_raw_eeg_snippet(args.hdf5_path, args.group_0, args.start_idx_0).to(device)
    eeg_1 = load_raw_eeg_snippet(args.hdf5_path, args.group_1, args.start_idx_1).to(device)

    # Bipolar Channels mapping
    channel_names = [
        "FP1-F7",
        "F7-T3",
        "T3-T5",
        "T5-O1",
        "FP2-F8",
        "F8-T4",
        "T4-T6",
        "T6-O2",
        "T3-C3",
        "C3-CZ",
        "CZ-C4",
        "C4-T4",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "A1-T3",
        "T4-A2"
    ]

    # 3. Calculate Saliency / CAM
    print(f"Calculating gradients for target class {args.target_class}...")
    
    # Conditional flip based on target class
    if args.target_class == 0:
        print("Target class is 0: Flipping eeg_0 and eeg_1 for model inference.")
        # We pass eeg_1 as the first input, and eeg_0 as the second.
        # The function returns (logits, cam_A, cam_B), so we unpack them as (logits, cam_1, cam_0)
        # to ensure the heatmaps correctly match the original timepoints during plotting.
        logits, cam_1, cam_0 = compute_eeg_saliency(
            combined_model, 
            eeg_1, 
            eeg_0, 
            location_embeddings, 
            args.target_class
        )
    else:
        print("Target class is 1: Keeping eeg_0 and eeg_1 in original order.")
        # Standard order
        logits, cam_0, cam_1 = compute_eeg_saliency(
            combined_model, 
            eeg_0, 
            eeg_1, 
            location_embeddings, 
            args.target_class
        )
    
    pred_prob = torch.softmax(torch.tensor(logits), dim=1)[0].numpy()
    print(f"Model Prediction Probabilities: Class 0: {pred_prob[0]:.4f}, Class 1: {pred_prob[1]:.4f}")

    # 4. Visualize and Save
    plot_eeg_with_cam(
        eeg_0.cpu().detach().numpy()[0], 
        cam_0, 
        channel_names, 
        title=f"Timepoint 0 ({args.group_0}) - Saliency Map, Class: {args.target_class}, Model Pred: [{pred_prob[0]:.4f}, {pred_prob[1]:.4f}]", 
        save_path="cam_timepoint_0.png"
    )

    plot_eeg_with_cam(
        eeg_1.cpu().detach().numpy()[0], 
        cam_1, 
        channel_names, 
        title=f"Timepoint 1 ({args.group_1}) - Saliency Map, Class: {args.target_class}, Model Pred: [{{pred_prob[0]:.4f}}, {pred_prob[1]:.4f}]", 
        save_path="cam_timepoint_1.png"
    )