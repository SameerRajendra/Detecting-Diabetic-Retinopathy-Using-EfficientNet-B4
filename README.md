# Detecting Diabetic Retinopathy — EfficientNet-B4

> A deep learning pipeline for 5-class **Diabetic Retinopathy severity grading** from retinal fundus images using **EfficientNet-B4** + CuPy GPU acceleration. Achieves **80.49% validation accuracy** and an F1-score of **0.91 on the “No DR” class**.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Requirements](#requirements)

---

## Overview

Diabetic Retinopathy (DR) is a leading cause of blindness, but early screening is limited by the scarcity of trained ophthalmologists. This model automates severity grading into 5 clinical stages:

| Grade | Label | Severity |
|-------|-------|----------|
| 0 | No DR | Healthy |
| 1 | Mild | Microaneurysms |
| 2 | Moderate | Hemorrhages |
| 3 | Severe | Venous beading |
| 4 | Proliferative | Neovascularization |

**Key design choices:**
- **EfficientNet-B4** backbone (380×380 input) for high-resolution retinal feature extraction
- **CuPy-powered data loader** (`dataset_utils.py`) for GPU-accelerated preprocessing
- **Balanced class weights** via `sklearn.compute_class_weight` injected into `nn.CrossEntropyLoss` to handle severe class imbalance (most patients have No DR)
- **Dropout (0.3)** on classifier head for regularization

---

## Key Results

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **80.49%** |
| No DR F1-score | **0.91** |
| Macro F1-score | ~0.51 |
| Architecture | EfficientNet-B4 |
| Input Resolution | 380 × 380 px |

> **Note on macro F1:** The macro score reflects the difficulty of minority classes (Grades 1–4), which are severely underrepresented in the dataset. The weighted average F1 is substantially higher. This is an inherent challenge in medical imaging datasets and is documented in the [full report](https://raw.githubusercontent.com/SameerRajendra/Detecting-Diabetic-Retinopathy-Using-EfficientNet-B4/main/Diabetic%20Retinopathy%20Detection%20with%20CNNs.pdf).

---

## Architecture

```text
  [ Retinal Fundus Image (JPEG) ]
              │  380×380 px
              ▼
  [ CuPy GPU Preprocessing (dataset_utils.py) ]
              │  Normalize + Augment
              ▼
  [ EfficientNet-B4 Backbone ]
              │  ImageNet pre-trained weights
              │  Transfer learning (fine-tuned)
              ▼
  [ Dropout (0.3) + FC Classifier ]
              │
              ▼
  [ Weighted CrossEntropyLoss ]
              │  sklearn balanced class weights
              ▼
  [ Grade 0–4 Severity Prediction ]
```

---

## Repository Structure

```
Detecting-Diabetic-Retinopathy-Using-EfficientNet-B4/
├── Detecting Diabetic Retinopathy Using EfficientNet-B4.ipynb  # Training + evaluation notebook
├── dataset_utils.py                                            # CuPy-powered GPU dataset loader
├── requirements.txt                                            # Python dependencies
├── Diabetic Retinopathy Detection with CNNs.pdf                # Full project report
└── README.md
```

---

## Setup & Installation

```bash
git clone https://github.com/SameerRajendra/Detecting-Diabetic-Retinopathy-Using-EfficientNet-B4.git
cd Detecting-Diabetic-Retinopathy-Using-EfficientNet-B4
pip install -r requirements.txt
```

---

## Dataset

Download from the Kaggle competition:

**[📥 Kaggle: Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)**

Expected folder structure after download:

```
data/
├── train/
│   ├── 10_left.jpeg
│   ├── 10_right.jpeg
│   └── ...
├── test/
│   └── ...
└── trainLabels.csv
```

---

## Usage

```bash
# Open the notebook and run cells sequentially
jupyter notebook "Detecting Diabetic Retinopathy Using EfficientNet-B4.ipynb"
```

The notebook covers:
1. Dataset loading with CuPy GPU preprocessing
2. EfficientNet-B4 model construction + transfer learning setup
3. Class-balanced training with `WeightedCrossEntropyLoss`
4. Evaluation: accuracy, F1-score, confusion matrix

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
cupy-cuda12x>=12.0.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
pandas>=2.0.0
albumentations>=1.3.0  # optional, for augmentation
```

---

*Course: Stevens Institute of Technology • Jan–May 2025*  
*Author: [Sameer Rajendra](https://github.com/SameerRajendra)*  
*[📄 Read the Full Report](https://raw.githubusercontent.com/SameerRajendra/Detecting-Diabetic-Retinopathy-Using-EfficientNet-B4/main/Diabetic%20Retinopathy%20Detection%20with%20CNNs.pdf)*
