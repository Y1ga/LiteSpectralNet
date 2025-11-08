## Overview

**LiteSpectralNet** is a lightweight neural network model designed for spectral data restoration. The model employs an encoder-decoder architecture combined with Residual Depthwise Separable Convolution (ResidualDW) and Channel Attention mechanism (SE Block), enabling efficient compression, restoration, and reconstruction of spectral information. 

### LSNet's Structure

![image-20251107213148390](C:\Users\z1002\AppData\Roaming\Typora\typora-user-images\image-20251107213148390.png)

### 1. Residual Depthwise Separable Convolution (ResidualDW)

```
Depthwise Conv (Group Convolution)
         ↓
    Pointwise Conv (1×1 Convolution)
         ↓
   Batch Normalization
         ↓
    Residual Connection + SiLU Activation
```

Significantly reduces computational complexity through the combination of group convolution and 1×1 convolution.

### 2. SE Attention Mechanism

Leverages interdependencies between channels to adaptively rescale feature maps:

```
Input → Global Average Pooling → FC Layers → Sigmoid → Channel Weighting
```

### 3. Encoder-Decoder Architecture

- **Encoder**: Progressively extracts high-level semantic features (3 layers of ResidualDW + SE blocks)
- **Decoder**: Progressively restores spatial resolution (Deconvolution + Interpolation + Restoration)

## Quick Start

### 1. Data Preparation

Download the dataset and place it in the `./data/` directory:

```bash
# Dataset download link
# https://pan.baidu.com/s/18Yiqq_UjjoH8ZyKADj3OqA?pwd=tdti
# Password: tdti

# Ensure the data file structure is as follows
data/
├── train_dataset.mat
├── ...
```

#### MATLAB File Format Requirements

Data should be in `.mat` format with the following variables:

- `input_data`: Input spectral data, shape `[num_channels, num_samples]`
- `output_data`: Target spectral data, shape `[num_channels, num_samples]`

### 2. Train the Model

```bash
python train.py
```

After training, the model will be saved to `./nets/lsnet.pth`

### 3. Inference with Pre-trained Model

```python
import torch
from model import LiteSpectralNet

# Load model
checkpoint = torch.load('./nets/lsnet.pth', map_location='cpu')
model = LiteSpectralNet(output_channel=checkpoint['output_channel'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    input_spectrum = torch.randn(1, input_wavelength_num)  # Single spectrum
    output_spectrum = model(input_spectrum)
    print(output_spectrum.shape)  # [1, output_wavelength_num]
```

## Dataset

### Download Link

- **BaiduPan**: https://pan.baidu.com/s/18Yiqq_UjjoH8ZyKADj3OqA?pwd=tdti
- **Password**: tdti
- `scene01.mat` represents the first biospecimen's spectral image (from 400-700nm and the interval is 10nm).
- `train_dataset.mat` represents the CAVE dataset input our hybrid encoding system.

