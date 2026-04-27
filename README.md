# InSeC: Steganalysis Model Based on Inter-Codeword Sensitivity Caption for Compressed Speech Streams

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://doi.org/10.1109/ACCESS.2024.3519094)
[![GitHub Stars](https://img.shields.io/github/stars/zhousandeqingshu/ZhCode?style=social)](https://github.com/zhousandeqingshu/ZhCode/stargazers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)

> **Official Implementation of *InSeC* – A deep learning framework for detecting Joint Parallel Steganography (JPS) in low-bit-rate VoIP compressed speech streams.**  
> *Published in IEEE Access, December 2024*

---

## 🔍 Overview

**InSeC** (Inter‑codeword Sensitivity Caption) is a novel steganalysis model designed to detect hidden information in **VoIP compressed speech**, particularly when multiple steganographic algorithms (e.g., **CNV** and **PSR**) are used simultaneously – a scenario known as **Joint Parallel Steganography (JPS)**. Unlike existing methods that target a single steganographic algorithm or suffer from poor generalization, InSeC introduces a **two‑stage neural architecture** that:

1. **Captures steganography‑sensitive codeword pairs** from multiple perspectives (feed‑forward, stacked LSTMs, and channel‑wise attention).
2. **Re‑perceives fine‑grained correlations** via a dedicated CNN module to amplify discriminative features.

Our method achieves **state‑of‑the‑art detection accuracy** across various embedding rates, speech durations, and packet loss conditions, while satisfying **real‑time requirements** (2.39 ms to process 1 second of speech).

---

## 🚀 Key Features

- **✅ Multi‑algorithm steganalysis** – Detects **CNV**, **PSR**, and their combination (**JPS**) in a single unified model.  
- **✅ High accuracy** – Outperforms four recent baselines by **9.07% ~ 25.27%** on JPS detection (see results below).  
- **✅ Robust to low embedding rates** – Maintains >50% accuracy even at **1% embedding rate** (substantially better than random).  
- **✅ Short‑segment support** – Works on speech frames as short as **0.1 seconds** (excellent for real‑time streaming).  
- **✅ Resilient to packet loss** – Over **81% accuracy** at 20% packet loss on JPS datasets.  
- **✅ Lightweight & fast** – Only **2.39 ms** per 1‑second sample, suitable for backbone router deployment.  

---

## 📐 Architecture

InSeC consists of three main modules:

1. **Steganography‑Sensitive Codeword‑Pair Caption Module (SCM)**  
   - Three parallel branches:  
     - *Feed‑Forward Fully Connected (FFC)* – models simple non‑linear mappings.  
     - *Stacked LSTMs* – captures both local and long‑range dependencies.  
     - *Channel Interaction Attention Module (CIAM)* – weights important feature channels.  
   - A fusion layer aggregates outputs from the three branches.

2. **Fine‑Grained Correlation Re‑Perception Module (FCRM)**  
   - Three 1D convolutional layers (kernel size 3) with batch normalization and ReLU.  
   - Average pooling + fully connected layer to produce a compact 32‑D feature vector.

3. **Feature Classification Module (FCM)**  
   - Two fully connected layers with dropout (p=0.5) and sigmoid activation.  
   - Outputs the probability of hidden information.

![InSeC Architecture](https://github.com/zhousandeqingshu/ZhCode/blob/main/InSeC.png)  
*Figure 2 from the paper – the full network structure.*

---

## 📊 Performance Highlights

### 📈 JPS Detection Accuracy (10‑second samples, English dataset)

| Embedding Rate | RNN‑SM | SFFN | CSW | SANet | **InSeC (Ours)** |
|---------------|--------|------|-----|-------|------------------|
| 10%           | 52.39% | 68.21% | 67.50% | 67.14% | **71.42%** |
| 30%           | 61.30% | 75.00% | 77.50% | 77.29% | **86.57%** |
| 50%           | 74.52% | 85.27% | 85.70% | 84.26% | **92.54%** |

> *InSeC consistently outperforms all baselines across low, medium, and high embedding rates.*

### ⏱️ Real‑time Detection Speed

| Method       | Time (ms per 1‑second speech) |
|--------------|-------------------------------|
| RNN‑SM       | 0.77                          |
| **InSeC**    | **2.39**                      |
| SFFN         | 2.49                          |
| SANet        | 3.27                          |
| CSW          | 3.99                          |

✅ **InSeC is among the fastest models** while maintaining the highest accuracy.

### 📦 Short‑Length Speech (0.1 s, JPS Chinese dataset)

| Method       | Accuracy |
|--------------|----------|
| RNN‑SM       | 52.07%   |
| SFFN         | 68.55%   |
| CSW          | 69.42%   |
| SANet        | 70.31%   |
| **InSeC**    | **86.99%** |

> *InSeC excels even on extremely short utterances, crucial for real‑time streaming interception.*

### 📉 Robustness to Packet Loss (30% embedding rate, JPS English)

| Packet Loss | RNN‑SM | SFFN | CSW | SANet | **InSeC** |
|-------------|--------|------|-----|-------|------------|
| 5%          | 73.22% | 79.40% | 81.21% | 82.75% | **88.25%** |
| 15%         | 71.01% | 77.34% | 80.11% | 81.23% | **84.74%** |

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
git clone https://github.com/zhousandeqingshu/ZhCode.git
cd ZhCode
pip install -r requirements.txt
```
### Training
```bash
python train.py --stego_type JPS --embed_rate 30 --frame_len 10 --epochs 50
```

### Testing
```bash
python test.py --model_path checkpoints/best_model.pth --data_path ./test_data
```

---
## 🗂️ Dataset Preparation

The model expects G.729 encoded speech in PCM format.

Use the dataset from [Lin et al. (2018)](https://example.com) containing:

- 41 hours of Chinese speech
- 72 hours of English speech

Generate cover/stego samples with CNV, PSR, or JPS embedding at desired rates (10%–100%).

Frame length can be set from 0.1 s to 2 s (default: 10 s for training).

## 🚀 Training

python train.py --stego_type JPS --embed_rate 30 --frame_len 10 --epochs 50


## 🧪 Testing

python test.py --model_path checkpoints/best_model.pth --data_path ./test_data



--- 
## 📄 Citation
If you find our work useful for your research, please cite:
``` bash
@article{Zhang2024InSeC,
  author    = {Hao Zhang and Jie Yang and Feipeng Gao and Jiacheng Yuan},
  title     = {InSeC: Steganalysis Model Based on Inter-Codeword Sensitivity Caption for Compressed Speech Streams},
  journal   = {IEEE Access},
  volume    = {12},
  pages     = {192252--192262},
  year      = {2024},
  doi       = {10.1109/ACCESS.2024.3519094},
  note      = {Accepted 9 December 2024, published 17 December 2024}
}

```

--- 
## 📜 License
This project is licensed under the MIT License – see the LICENSE file for details.


--- 
## 🤝 Contributing
Issues and pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---
## 📧 Contact
- First author: Zhang Hao – 3181280664@qq.com
- Code repository: https://github.com/zhousandeqingshu/ZhCode

--- 
## ⭐ Star History
If InSeC has helped your research, a ⭐ on GitHub would be an enormous encouragement to us and helps more people find this work！
https://api.star-history.com/svg?repos=zhousandeqingshu/ZhCode&type=Date
