import os
import glob
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from asn_model import ASNConfig, AdaptiveSpikingNetwork

# =====================================================================
# CONFIGURATION & PARAMETERS
# =====================================================================
DATA_DIR = "./dataset"      # Path to root dataset directory
LOAD_WEIGHTS_PATH = None    # Path e.g. "asn_weights.npz" or None
SAVE_WEIGHTS_PATH = "asn_weights.npz"

CONFIG = ASNConfig(
    # Image Geometry
    img_height=100,           # Input image height (pixels)
    img_width=100,            # Input image width (pixels)
    
    # Temporal & Training Loop
    time_steps=20,            # Simulation duration per sample (timesteps)
    epochs=1,                 # Training passes over full dataset
    
    # SNN Architecture
    n_neurons=100,            # Spiking neurons per group
    
    # Gabor Feature Bank
    gabor_scales=6,           # Wavelength/frequency variations
    gabor_angles=8,           # Orientation angle divisions
    gabor_phases=1,           # Phase shifts (1=cosine, 2=sine+cosine)
    gabor_ksize=11,           # Filter kernel grid size (pixels)
    
    # Patch Slicing
    patch_height=10,          # Sub-window patch height (pixels)
    patch_width=10,           # Sub-window patch width (pixels)
    patch_step=10,            # Patch sliding stride (pixels)
    
    # Bio-Spiking Dynamics
    alpha=0.001,              # Threshold adaptation learning rate
    beta=0.1,                 # Membrane leak/decay factor
    gamma=0.01,               # Plastic synaptic weight update rate
    weight_decay=0.001,       # Multiplicative synaptic weight decay factor
    p_target=0.05,            # Target neuron firing rate (5%)
    initial_thresh_val=5.0,   # Starting firing threshold (mV)
    min_thresh_val=1.0        # Minimum threshold lower floor (mV)
)


# =====================================================================
# DYNAMIC PURE NUMPY GABOR FEATURE EXTRACTION
# =====================================================================
def generate_gabor_bank_numpy(
    gabor_scales: int = 6, 
    gabor_angles: int = 8, 
    gabor_phases: int = 1, 
    gabor_ksize: int = 11
) -> list:
    """
    Generates Gabor filter bank dynamically in pure NumPy.
    Total extracted channels = gabor_scales * gabor_angles * gabor_phases.
    """
    kernels = []
    limit = gabor_ksize // 2
    y, x = np.mgrid[-limit:limit+1, -limit:limit+1]
    
    wavelengths = [2 + i for i in range(gabor_scales)]
    angles = np.arange(0, np.pi, np.pi / gabor_angles)
    phases = np.linspace(0, np.pi, gabor_phases, endpoint=False)
    
    for lambd in wavelengths:
        sigma = 0.5 * lambd
        for theta in angles:
            for psi in phases:
                x_theta = x * np.cos(theta) + y * np.sin(theta)
                y_theta = -x * np.sin(theta) + y * np.cos(theta)
                
                envelope = np.exp(-(x_theta**2 + 0.25 * y_theta**2) / (2 * sigma**2))
                wave = np.cos(2 * np.pi * x_theta / lambd + psi)
                kernel = envelope * wave
                
                # Zero-mean DC component removal
                kernel -= np.mean(kernel)
                kernels.append(kernel)
                
    return kernels


def convolve2d_spatial(image: np.ndarray, kernel: np.ndarray, config: ASNConfig) -> np.ndarray:
    """
    Performs 2D spatial convolution with dynamic padding to guarantee 
    output spatial dimensions match (config.img_height, config.img_width).
    """
    in_h, in_w = image.shape
    kh, kw = kernel.shape
    
    out_h, out_w = config.img_height, config.img_width
    
    # Calculate required padding across spatial dimensions
    pad_h_total = max(0, out_h - in_h + kh - 1)
    pad_w_total = max(0, out_w - in_w + kw - 1)
    
    # Distribute padding across borders
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left
    
    # Safety fallback if spatial dimensions are smaller than kernel dimensions
    pad_mode = 'reflect' if min(in_h, in_w) > max(kh, kw) else 'edge'
    
    padded_img = np.pad(
        image, 
        ((pad_top, pad_bottom), (pad_left, pad_right)), 
        mode=pad_mode
    )
    
    windows = sliding_window_view(padded_img, window_shape=(kh, kw))
    return np.sum(windows * np.flip(kernel), axis=(-2, -1))


# =====================================================================
# DATA LOADING & PREPROCESSING
# =====================================================================
def load_dataset_from_dir(data_dir: str, config: ASNConfig) -> list:
    """
    Loads, resizes, and auto-pads images recursively across top-level and 
    sub-directories to extract Gabor channel tensors with strict Min-Max scaling.
    """
    valid_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(
            glob.glob(os.path.join(data_dir, "**", ext), recursive=True)
        )

    if not image_paths:
        raise FileNotFoundError(f"No valid image files found in directory tree: '{data_dir}'")

    samples = []
    print(f"[+] Loading and processing {len(image_paths)} images recursively from '{data_dir}'...")

    target_spatial = (config.img_width, config.img_height)
    
    gabor_bank = generate_gabor_bank_numpy(
        gabor_scales=config.gabor_scales,
        gabor_angles=config.gabor_angles,
        gabor_phases=config.gabor_phases,
        gabor_ksize=config.gabor_ksize
    )

    for path in image_paths:
        img = Image.open(path).convert("L")
        img = img.resize(target_spatial, Image.Resampling.LANCZOS)
        img_np = np.array(img, dtype=np.float32) / 255.0

        filtered_channels = []
        for kernel in gabor_bank:
            filtered = convolve2d_spatial(img_np, kernel, config=config)
            filtered = np.maximum(0, filtered)  # Rectification
            filtered_channels.append(filtered)

        img_multichannel = np.stack(filtered_channels, axis=0)
        
        # Strict Min-Max scaling to [0.0, 1.0] with zero-division safeguard
        min_val = img_multichannel.min()
        max_val = img_multichannel.max()

        if max_val > min_val:
            img_multichannel = (img_multichannel - min_val) / (max_val - min_val)
        else:
            img_multichannel = np.zeros_like(img_multichannel)

        # Explicit bound clipping to protect Poisson encoding
        img_multichannel = np.clip(img_multichannel, 0.0, 1.0)
        samples.append(img_multichannel)

    return samples


# =====================================================================
# EXECUTION PIPELINE
# =====================================================================
def main():
    model = AdaptiveSpikingNetwork(CONFIG)

    if LOAD_WEIGHTS_PATH and os.path.exists(LOAD_WEIGHTS_PATH):
        model.load_weights(LOAD_WEIGHTS_PATH)

    if DATA_DIR and os.path.exists(DATA_DIR):
        samples = load_dataset_from_dir(DATA_DIR, CONFIG)
    else:
        print("[!] No valid DATA_DIR provided or folder empty. Running pipeline with synthetic test tensors...")
        samples = [
            np.random.rand(CONFIG.patch_channels, CONFIG.img_height, CONFIG.img_width)
            for _ in range(2)
        ]

    print(f"[+] Starting SNN execution (Groups: {CONFIG.n_groups}, Channels: {CONFIG.patch_channels}, Kernel: {CONFIG.gabor_ksize}x{CONFIG.gabor_ksize})...")
    history = model.train_core(samples)

    if SAVE_WEIGHTS_PATH:
        model.save_weights(SAVE_WEIGHTS_PATH)


if __name__ == "__main__":
    main()