import os
import argparse
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import OrderedDict

# Import LaBraM models and timm
from timm.models import create_model

import sys
# Add the directory containing your repositories
sys.path.append("/home/yhchoi")

# Import LILIE
from LILIE.models.models import LILIE  # Adjust import path if needed
from LaBraM.modeling_finetune import labram_base_patch200_200
import LaBraM.data_processor as data_processor
import LaBraM.utils as utils

class EndToEndLongitudinal(torch.nn.Module):
    """
    Wraps LaBraM and LILIE together to allow gradients to flow 
    from the final classification output all the way back to the raw EEG input.
    """
    def __init__(self, labram_model, lilie_model, segment_size=200):
        super().__init__()
        self.labram = labram_model
        self.lilie = lilie_model
        self.segment_size = segment_size
        
        # Ensure models are in eval mode to disable dropout/batchnorm
        self.labram.eval()
        self.lilie.eval()

    def forward(self, eeg_0, eeg_1, input_chans):
        # 1. Reshape raw temporal data to LaBraM's expected patch format
        # Current shape: (B, C, T) -> Target shape: (B, C, Num_Segments, Segment_Size)
        B, C, T = eeg_0.shape
        num_segments = T // self.segment_size
        
        eeg_0_patched = eeg_0.view(B, C, num_segments, self.segment_size)
        eeg_1_patched = eeg_1.view(B, C, num_segments, self.segment_size)

        # 2. Extract embeddings using LaBraM
        embed_0 = self.labram(eeg_0_patched, input_chans=input_chans, return_all_tokens=True)
        embed_1 = self.labram(eeg_1_patched, input_chans=input_chans, return_all_tokens=True)
        
        print(embed_0.shape)
        # 3. Pool and classify using LILIE
        pooled = self.lilie.pooler(embed_0, embed_1)
        logits = self.lilie.clf(pooled)
        return logits

def load_raw_eeg_snippet(hdf5_path, group_name, start_idx, window_size=7600):
    """Loads a specific raw EEG snippet from the HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        data = f[group_name]['eeg'][:, start_idx : start_idx + window_size]
        data = data.astype(np.float32)
    return torch.from_numpy(data).unsqueeze(0) # Add batch dimension: (1, Channels, Time)

def compute_eeg_saliency(model, eeg_0, eeg_1, input_chans, target_class=1):
    """
    Computes the smoothed input gradients for both longitudinal EEG inputs.
    """
    # Enable gradient tracking on raw continuous inputs
    eeg_0.requires_grad = True
    eeg_1.requires_grad = True

    # Forward pass through the stitched model
    logits = model(eeg_0, eeg_1, input_chans)
    
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

def load_labram_checkpoint(model, ckpt_path):
    """Loads weights specifically for LaBraM filtering out student/teacher keys if necessary."""
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    checkpoint_model = None
    for model_key in ['model|module', 'model', 'state_dict']:
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            break
            
    if checkpoint_model is None:
        checkpoint_model = checkpoint
        
    # Filter 'student.' prefix if it exists
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('student.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
            
    # Remap pooling norm if necessary
    if 'norm.weight' in new_dict and 'fc_norm.weight' not in new_dict:
        new_dict['fc_norm.weight'] = new_dict['norm.weight']
        new_dict['fc_norm.bias'] = new_dict['norm.bias']

    utils.load_state_dict(model, new_dict, prefix='')
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM / Saliency Maps for LILIE Model with LaBraM")
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to HDF5 data')
    parser.add_argument('--labram_ckpt', type=str, required=True, help='Path to trained LaBraM checkpoint')
    parser.add_argument('--lilie_ckpt', type=str, required=True, help='Path to trained LILIE lightning checkpoint (.ckpt)')
    parser.add_argument('--target_class', type=int, default=1, help='Class index to calculate gradients for (0 or 1)')
    
    # LaBraM Configuration
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, help='Name of LaBraM model')
    parser.add_argument('--window_size', type=int, default=7600, help='Total EEG window size to load from HDF5')
    parser.add_argument('--patch_size', type=int, default=200, help='Patch/Segment size for LaBraM')
    parser.add_argument('--labram_embed_dim', type=int, default=768, help='Output embedding dimension of LaBraM')

    # Parameters for the specific sample
    parser.add_argument('--group_0', type=str, required=True, help='HDF5 Group name for Timepoint 0')
    parser.add_argument('--start_idx_0', type=int, default=0, help='Start index for Timepoint 0')
    parser.add_argument('--group_1', type=str, required=True, help='HDF5 Group name for Timepoint 1')
    parser.add_argument('--start_idx_1', type=int, default=0, help='Start index for Timepoint 1')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure window is divisible by patch_size for LaBraM reshaping
    if args.window_size % args.patch_size != 0:
        raise ValueError(f"window_size ({args.window_size}) must be cleanly divisible by patch_size ({args.patch_size})")

    # 1. Setup Models
    print("Loading LaBraM model...")
    labram_model = create_model( # Set to defaults
        args.model,
        pretrained=False,
        num_classes=0, # Force 0 for embedding extraction
        drop_rate=0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
        EEG_size=3200,
    )
    labram_model = load_labram_checkpoint(labram_model, args.labram_ckpt)

    print("Loading LILIE model...")
    lilie_model = LILIE.load_from_checkpoint(args.lilie_ckpt, 
                                             input_dim=args.labram_embed_dim, 
                                             embedding_size=128, 
                                             num_classes=2, 
                                             pool_method="Attentive", 
                                             clf_method="NN")
    
    combined_model = EndToEndLongitudinal(labram_model, lilie_model, segment_size=args.patch_size).to(device)

    # 2. Load Data & Input Channels
    eeg_0 = load_raw_eeg_snippet(args.hdf5_path, args.group_0, args.start_idx_0, window_size=args.window_size).to(device)
    eeg_1 = load_raw_eeg_snippet(args.hdf5_path, args.group_1, args.start_idx_1, window_size=args.window_size).to(device)

    # Define channel names
    channel_names_caps = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", 
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", 
    "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2"
    ]

    channel_names = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", 
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", 
    "A1", "A2", "Fz", "Cz", "Pz", "T1", "T2"
    ]

    # Generate input channel representations native to LaBraM
    input_chans = utils.get_input_chans(channel_names_caps)

    # 3. Calculate Saliency / CAM
    print(f"Calculating gradients for target class {args.target_class}...")
    
    # Conditional flip based on target class
    if args.target_class == 1:
        print("Target class is 1: Flipping eeg_0 and eeg_1 for model inference.")
        logits, cam_1, cam_0 = compute_eeg_saliency(
            combined_model, 
            eeg_1, 
            eeg_0, 
            input_chans, 
            args.target_class
        )
    else:
        print("Target class is 0: Keeping eeg_0 and eeg_1 in original order.")
        logits, cam_0, cam_1 = compute_eeg_saliency(
            combined_model, 
            eeg_0, 
            eeg_1, 
            input_chans, 
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
        title=f"Timepoint 1 ({args.group_1}) - Saliency Map, Class: {args.target_class}, Model Pred: [{pred_prob[0]:.4f}, {pred_prob[1]:.4f}]", 
        save_path="cam_timepoint_1.png"
    )