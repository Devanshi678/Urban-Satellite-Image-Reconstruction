# Reconstruction of Urban Satellite Imagery Using Deep Learning  
### Baseline CNN Models and GAN-Based Framework

## 📌 Project Overview
This project investigates the reconstruction of high-quality urban satellite imagery from degraded inputs using deep learning. The goal is to recover fine structural details such as buildings, roads, and urban layouts while preserving spatial consistency and visual realism.

Two reconstruction strategies were implemented and compared:

1. **Baseline convolutional models**
   - U-Net
   - Residual U-Net
2. **GAN-based reconstruction framework**
   - U-Net-based generator
   - CNN-based discriminator

The project emphasizes both **quantitative accuracy** and **qualitative visual quality**, highlighting the limitations of pixel-wise reconstruction losses and the advantages of adversarial learning.

This work was completed as part of a graduate-level **Deep Learning course** and is intended for **educational and research purposes**.

---

## 🧠 What I Did in This Project
- Designed and implemented **U-Net and Residual U-Net** models for supervised image reconstruction
- Built a **GAN framework** to improve perceptual quality of reconstructed satellite images
- Developed **separate preprocessing pipelines** for baseline and GAN models
- Trained models on **preprocessed urban satellite imagery**
- Evaluated results using **quantitative metrics** and **visual reconstruction maps**
- Compared baseline and GAN outputs to analyze trade-offs between smoothness and realism

---

## 📂 Dataset Used in This Project

### Dataset Description
The dataset consists of **urban satellite images** organized as paired data:
- **Input images**: degraded or low-quality satellite imagery
- **Ground truth images**: high-quality reference images

These images represent urban regions containing:
- Buildings
- Roads
- Dense city infrastructure

### ⚠️ Dataset Availability
> **The dataset is NOT included in this repository** due to size and licensing restrictions.

The dataset was obtained from an **academic / course-provided satellite imagery source** and was manually preprocessed before training.

---

## ⚙️ Preprocessing
All preprocessing steps are documented and reproducible.

Preprocessing includes:
- Image resizing and normalization
- Pairing degraded and ground-truth images
- Train / validation / test split creation

📓 Preprocessing code can be found in:
Baseline/Preprocessing/
GAN/Preprocessing/


These notebooks and scripts document the full data-cleaning pipeline **without including the raw data**.

---

## 📊 Results and Observations

### 🔹 Baseline Models (U-Net / Residual U-Net)
- Produce smooth reconstructions
- Preserve overall structure
- Tend to blur fine urban details
- Perform well on pixel-wise metrics

### 🔹 GAN-Based Model
- Produces sharper reconstructions
- Better preserves edges and textures
- Improves perceptual realism
- Introduces adversarial stability challenges during training

### 🔹 Visual Outputs
The project report includes:
- Reconstructed satellite images
- Difference maps highlighting reconstruction errors
- Visual comparisons between:
  - Input image
  - Baseline reconstruction
  - GAN reconstruction
  - Ground truth

📁 Sample output images and reconstruction maps are available in:
- results/

  


