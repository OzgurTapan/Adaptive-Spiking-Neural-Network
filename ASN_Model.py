import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from dataclasses import dataclass


@dataclass
class ASNConfig:
    # Image Geometry
    img_height: int = 100           # Input image height (pixels)
    img_width: int = 100            # Input image width (pixels)
    
    # Temporal & Training Loop
    time_steps: int = 20            # Simulation duration per sample (timesteps)
    epochs: int = 1                 # Training passes over full dataset
    
    # SNN Architecture
    n_neurons: int = 100            # Spiking neurons per group
    
    # Gabor Feature Bank
    gabor_scales: int = 6           # Wavelength/frequency variations
    gabor_angles: int = 8           # Orientation angle divisions
    gabor_phases: int = 1           # Phase shifts (1=cosine, 2=sine+cosine)
    gabor_ksize: int = 11           # Filter kernel grid size (pixels)
    
    # Patch Slicing
    patch_height: int = 10          # Sub-window patch height (pixels)
    patch_width: int = 10           # Sub-window patch width (pixels)
    patch_step: int = 10            # Patch sliding stride (pixels)
    
    # Bio-Spiking Dynamics
    alpha: float = 0.001            # Threshold adaptation learning rate
    beta: float = 0.1               # Membrane leak/decay factor
    gamma: float = 0.01             # Plastic synaptic weight update rate
    weight_decay: float = 0.001     # Multiplicative synaptic weight decay factor
    p_target: float = 0.05          # Target neuron firing rate (5%)
    initial_thresh_val: float = 5.0 # Starting firing threshold (mV)
    min_thresh_val: float = 1.0     # Minimum threshold lower floor (mV)

    def __post_init__(self):
        if self.patch_height > self.img_height or self.patch_width > self.img_width:
            raise ValueError("Patch dimensions cannot exceed total image dimensions.")

    @property
    def patch_channels(self) -> int:
        """Total input feature channels derived from Gabor filter bank setup."""
        return self.gabor_scales * self.gabor_angles * self.gabor_phases

    @property
    def n_groups(self) -> int:
        """Dynamically calculates total spatial patch groups across the image grid."""
        n_y = (self.img_height - self.patch_height) // self.patch_step + 1
        n_x = (self.img_width - self.patch_width) // self.patch_step + 1
        return n_y * n_x


class AdaptiveSpikingNetwork:
    """
    Modular Adaptive Spiking Neural Network (SNN) using LIF dynamics,
    adaptive firing thresholds, and local plastic updates in pure NumPy.
    """
    def __init__(self, config: ASNConfig):
        self.config = config
        
        # Flattened spatial patch footprint across all feature channels
        self.patch_size = (
            self.config.patch_channels * 
            self.config.patch_height * 
            self.config.patch_width
        )
        
        # Synaptic weight matrix: (n_groups, n_neurons, patch_size)
        self.weights = np.random.uniform(
            0.1, 0.5, 
            size=(self.config.n_groups, self.config.n_neurons, self.patch_size)
        ).astype(np.float32)
        
        # Dynamic firing threshold array per neuron
        self.thresholds = np.full(
            (self.config.n_groups, self.config.n_neurons), 
            self.config.initial_thresh_val, 
            dtype=np.float32
        )

    def extract_patches(self, img_tensor: np.ndarray) -> np.ndarray:
        """
        Slices multi-channel feature maps into flattened patch vectors using 
        zero-copy vectorized sliding windows.
        Output shape: (n_groups, patch_size)
        """
        c, h, w = img_tensor.shape
        
        # Extract 3D sliding windows: shape (1, n_y_full, n_x_full, C, patch_h, patch_w)
        windows = sliding_window_view(
            img_tensor, 
            window_shape=(c, self.config.patch_height, self.config.patch_width)
        )[0]
        
        # Apply spatial stride along height and width
        strided_windows = windows[::self.config.patch_step, ::self.config.patch_step]
        n_y, n_x = strided_windows.shape[0], strided_windows.shape[1]
        
        # Reshape directly into (n_groups, patch_size)
        return strided_windows.reshape(n_y * n_x, -1).astype(np.float32)

    def train_core(self, samples: list) -> dict:
        """
        Simulates network dynamics across timesteps with homeostatic and synaptic updates.
        """
        history = {"mean_spikes": [], "mean_thresholds": []}

        for epoch in range(self.config.epochs):
            total_spikes = 0
            
            for sample_idx, img_tensor in enumerate(samples):
                # Extract patches and enforce strict Poisson probability bounds [0.0, 1.0]
                raw_patches = self.extract_patches(img_tensor)
                patches = np.clip(raw_patches, 0.0, 1.0)
                
                # Reset LIF membrane potentials
                v_mem = np.zeros(
                    (self.config.n_groups, self.config.n_neurons), 
                    dtype=np.float32
                )
                sample_spike_count = 0
                
                for t in range(self.config.time_steps):
                    # Rate-coded Poisson spike generation from continuous inputs
                    input_spikes = (
                        np.random.rand(*patches.shape) < patches
                    ).astype(np.float32)
                    
                    # Compute total synaptic drive per neuron
                    currents = np.einsum('gp, gnp -> gn', input_spikes, self.weights)
                    
                    # Leaky Integrate-and-Fire membrane update
                    v_mem = (1.0 - self.config.beta) * v_mem + currents
                    
                    # Output spike generation
                    spikes = (v_mem >= self.thresholds).astype(np.float32)
                    sample_spike_count += np.sum(spikes)
                    
                    # Reset membrane potential on firing
                    v_mem[spikes > 0] = 0.0
                    
                    # Homeostatic adaptive threshold update with lower bound floor
                    self.thresholds += self.config.alpha * (
                        spikes - self.config.p_target
                    )
                    self.thresholds = np.maximum(self.config.min_thresh_val, self.thresholds)
                    
                    # Plastic synaptic weight update with multiplicative decay (LTD)
                    dw = self.config.gamma * np.einsum('gn, gp -> gnp', spikes, input_spikes)
                    self.weights = (1.0 - self.config.weight_decay) * self.weights + dw
                    self.weights = np.clip(self.weights, 0.0, 1.0)

                total_spikes += sample_spike_count

            avg_spikes = total_spikes / max(1, len(samples))
            history["mean_spikes"].append(avg_spikes)
            history["mean_thresholds"].append(float(np.mean(self.thresholds)))
            
            print(f"Epoch {epoch + 1}/{self.config.epochs} | Avg Spikes/Sample: {avg_spikes:.2f} | Mean Threshold: {np.mean(self.thresholds):.3f}")

        return history

    def save_weights(self, filepath: str):
        """Saves weights and thresholds to a compressed archive."""
        np.savez_compressed(
            filepath, 
            weights=self.weights, 
            thresholds=self.thresholds
        )
        print(f"[+] Saved model parameters to '{filepath}'")

    def load_weights(self, filepath: str):
        """Loads weights and thresholds from a compressed archive with shape validation."""
        data = np.load(filepath)
        assert data["weights"].shape == self.weights.shape, \
            f"Weight shape mismatch: Loaded file {data['weights'].shape} vs Config {self.weights.shape}"
        self.weights = data["weights"]
        self.thresholds = data["thresholds"]
        print(f"[+] Loaded model parameters from '{filepath}'")