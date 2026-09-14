# Adaptive Spiking Neural Network (SNN) with Gabor Feature Extraction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy%20Only-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains a pure NumPy implementation of a modular Adaptive Spiking Neural Network (SNN) designed for visual feature processing. Built entirely without external deep learning frameworks, this codebase demonstrates foundational computer vision and biologically-inspired machine learning mechanics.

---

## 📐 Core Architecture
* **Pure NumPy Implementation:** The entire pipeline, from feature extraction to synaptic weight updates, is optimized using vectorized NumPy operations without reliance on external machine learning libraries.
* **Dynamic Gabor Filter Bank:** Generates a custom Gabor feature bank dynamically to extract spatial frequencies and orientations.
* **Leaky Integrate-and-Fire (LIF) Dynamics:** Neurons simulate biological membrane potential leak and integrate input currents over time to generate rate-coded Poisson spikes.
* **Homeostatic Plasticity & Learning:** Features adaptive firing thresholds to maintain target firing rates alongside plastic synaptic weight updates with multiplicative decay.
* **Custom Spatial Convolution:** Implements 2D spatial convolution with dynamic edge reflection and padding to preserve strict spatial dimensions during feature extraction.

---


## 🛠️ Repository Structure & Prerequisites
* `ASN_Main.py`: The main execution pipeline that handles recursive image discovery, strict Min-Max scaling preprocessing, bounding clips to protect Poisson encoding, and the training loop.
* `ASN_Model.py`: Contains the `ASNConfig` dataclass for hyperparameter management and the core `AdaptiveSpikingNetwork` class, which handles LIF membrane updates, threshold adaptations, and zero-copy sliding window patch extraction.
* numpy (For matrix operations and network math)
* Pillow (PIL) (For image loading and spatial resampling)

---

## ⚙️ Execution & Usage

1. To run the pipeline, ensure both `ASN_Main.py` and `ASN_Model.py` are saved in the same directory, along with an optional `./dataset` folder.
(supported image extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`)
(nested subfolders are automatically scanned recursively)
2. Execute the main script to process the images and begin the SNN training simulation:
```
python ASN_Main.py
```
3. If the dataset directory is empty or missing, the pipeline automatically falls back to running synthetic test tensors to ensure execution.
4. Upon completion, the network saves the learned synaptic weights and adaptive firing thresholds to a compressed archive named asn_weights.npz.
5. To continue using your trained weights in ASN_Main.py, you can configure the file to automatically load them at startup by setting the LOAD_WEIGHTS_PATH variable under the configuration & parameters.  
* Here is how you update the top configuration section in ASN_Main.py:  
DATA_DIR = "./dataset"      # Path to root dataset directory. 
LOAD_WEIGHTS_PATH = "asn_weights.npz"    # Set this to your saved weights file path  
SAVE_WEIGHTS_PATH = "asn_weights.npz"    # Path where updated weights will be saved  
* Once loaded, the network resumes its training cycle or inference passes using the pre-existing synaptic weights and adaptive firing thresholds.  

---

## 🎛️ Configuration Parameters (`ASNConfig`)

All network geometry, temporal steps, gabor configurations, and bio-spiking learning rates are centrally managed via the ASNConfig class, which can be modified inside ASN_Main.py. Below is the complete breakdown of every parameter available for tuning:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| **`img_height`** | `100` | Target input height of the image in pixels. |
| **`img_width`** | `100` | Target input width of the image in pixels. |
| **`time_steps`** | `20` | Duration of the simulation window per sample in timesteps. |
| **`epochs`** | `1` | Total number of training passes over the full dataset. |
| **`n_neurons`** | `100` | Number of independent spiking neurons allocated per spatial patch group. |
| **`gabor_scales`** | `6` | Number of wavelength/frequency variations in the Gabor filter bank. |
| **`gabor_angles`** | `8` | Number of orientation angle divisions across the filter bank. |
| **`gabor_phases`** | `1` | Phase shift divisions (1 = cosine, 2 = cosine + sine). |
| **`gabor_ksize`** | `11` | Spatial grid dimensions (in pixels) for the square Gabor kernels. |
| **`patch_height`** | `10` | Height dimension of local image sub-window patches. |
| **`patch_width`** | `10` | Width dimension of local image sub-window patches. |
| **`patch_step`** | `10` | Sliding stride step size across spatial dimensions. |
| **`alpha`** | `0.001` | Learning rate governing homeostatic threshold adaptation. |
| **`beta`** | `0.1` | Membrane potential leak/decay factor per timestep. |
| **`gamma`** | `0.01` | Plastic synaptic weight update rate (Hebbian learning rate). |
| **`weight_decay`** | `0.001` | Multiplicative synaptic weight decay factor (LTD component). |
| **`p_target`** | `0.05` | Target biological firing rate for homeostatic regulation (5%). |
| **`initial_thresh_val`** | `5.0` | Starting firing threshold barrier for newly initialized neurons. |
| **`min_thresh_val`** | `1.0` | Minimum threshold lower floor preventing homeostatic collapse. |
