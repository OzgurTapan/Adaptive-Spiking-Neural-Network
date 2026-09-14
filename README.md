# Adaptive Spiking Neural Network (SNN) with Gabor Feature Extraction

This repository contains a pure NumPy implementation of a modular Adaptive Spiking Neural Network (SNN) designed for visual feature processing. Built entirely without external deep learning frameworks, this codebase demonstrates foundational computer vision and biologically-inspired machine learning mechanics.

## Core Architecture
* **Pure NumPy Implementation:** The entire pipeline, from feature extraction to synaptic weight updates, is optimized using vectorized NumPy operations without reliance on external machine learning libraries[cite: 1, 2].
* **Dynamic Gabor Filter Bank:** Generates a custom Gabor feature bank dynamically to extract spatial frequencies and orientations.
* **Leaky Integrate-and-Fire (LIF) Dynamics:** Neurons simulate biological membrane potential leak and integrate input currents over time to generate rate-coded Poisson spikes[cite: 2].
* **Homeostatic Plasticity & Learning:** Features adaptive firing thresholds to maintain target firing rates alongside plastic synaptic weight updates with multiplicative decay[cite: 2].
* **Custom Spatial Convolution:** Implements 2D spatial convolution with dynamic edge reflection and padding to preserve strict spatial dimensions during feature extraction.

## Repository Structure
* `ASN_Main.py`: The main execution pipeline that handles recursive image discovery, strict Min-Max scaling preprocessing, bounding clips to protect Poisson encoding, and the training loop[cite: 1].
* `ASN_Model.py`: Contains the `ASNConfig` dataclass for hyperparameter management and the core `AdaptiveSpikingNetwork` class, which handles LIF membrane updates, threshold adaptations, and zero-copy sliding window patch extraction[cite: 2].

---

## Configuration Parameters (`ASNConfig`)

All network geometry, temporal steps, Gabor configurations, and bio-spiking learning rates are centrally managed via the `ASNConfig` class[cite: 1, 2]. Below is the complete breakdown of every parameter available for tuning:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| **`img_height`** | `100` | Target input height of the image in pixels[cite: 1, 2]. |
| **`img_width`** | `100` | Target input width of the image in pixels[cite: 1, 2]. |
| **`time_steps`** | `20` | Duration of the simulation window per sample in timesteps[cite: 1, 2]. |
| **`epochs`** | `1` | Total number of training passes over the full dataset[cite: 1, 2]. |
| **`n_neurons`** | `100` | Number of independent spiking neurons allocated per spatial patch group[cite: 1, 2]. |
| **`gabor_scales`** | `6` | Number of wavelength/frequency variations in the Gabor filter bank[cite: 1, 2]. |
| **`gabor_angles`** | `8` | Number of orientation angle divisions across the filter bank[cite: 1, 2]. |
| **`gabor_phases`** | `1` | Phase shift divisions (1 = cosine, 2 = cosine + sine)[cite: 1, 2]. |
| **`gabor_ksize`** | `11` | Spatial grid dimensions (in pixels) for the square Gabor kernels[cite: 1, 2]. |
| **`patch_height`** | `10` | Height dimension of local image sub-window patches[cite: 1, 2]. |
| **`patch_width`** | `10` | Width dimension of local image sub-window patches[cite: 1, 2]. |
| **`patch_step`** | `10` | Sliding stride step size across spatial dimensions[cite: 1, 2]. |
| **`alpha`** | `0.001` | Learning rate governing homeostatic threshold adaptation[cite: 1, 2]. |
| **`beta`** | `0.1` | Membrane potential leak/decay factor per timestep[cite: 1, 2]. |
| **`gamma`** | `0.01` | Plastic synaptic weight update rate (Hebbian learning rate)[cite: 1, 2]. |
| **`weight_decay`** | `0.001` | Multiplicative synaptic weight decay factor (LTD component)[cite: 1, 2]. |
| **`p_target`** | `0.05` | Target biological firing rate for homeostatic regulation (5%)[cite: 1, 2]. |
| **`initial_thresh_val`** | `5.0` | Starting firing threshold barrier for newly initialized neurons[cite: 1, 2]. |
| **`min_thresh_val`** | `1.0` | Minimum threshold lower floor preventing homeostatic collapse[cite: 1, 2]. |

---

## Execution & Usage

1. Ensure your images (supported extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`) are placed inside the `./dataset` root directory (nested subfolders are automatically scanned recursively)[cite: 1].
2. Execute the main script to process the images and begin the SNN training simulation[cite: 1]:
   ```bash
   python ASN_Main.py
If the dataset directory is empty or missing, the pipeline automatically falls back to running synthetic test tensors to ensure execution[cite: 1].
Upon completion, the network saves the learned synaptic weights and adaptive firing thresholds to a compressed archive named asn_weights.npz[cite: 1, 2].
Dependencies
numpy
[cite: 1, 2]
Pillow (PIL) for image loading and Lanczos spatial resampling[cite: 1]
