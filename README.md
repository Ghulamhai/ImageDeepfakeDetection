# Deepfake Detection with ViXNet and EfficientNet-B3

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Framework](https://img.shields.io/badge/Framework-PyTorch_&_Flask-orange.svg)

This project provides an efficient and powerful deepfake detection system based on the **ViXNet** architecture. This implementation replaces the original Xception backbone with a more lightweight **EfficientNet-B3**, creating a model optimized for performance on resource-constrained devices.

The model is trained on the **FaceForensics++ (FF++)** dataset and validated for generalization on the **Celeb-DF (CeDF)** dataset. A clean, modern web application built with Flask allows for easy, real-time inference.

![Deepfake Detection Web App](https://raw.githubusercontent.com/Ghulam-Haider/Deepfake-Detection-with-ViXNet-and-EfficientNet-B3/main/assets/webapp_screenshot.png)

---

## 🚀 Key Features

* **Advanced Dual-Branch Architecture**: Combines a patch-based Vision Transformer (ViT-B_16) for global context and an EfficientNet-B3 for local spatial feature extraction.
* **High-Performance Detection**: Achieves **~94% accuracy** on the FF++ dataset and demonstrates strong generalization with **~70% accuracy** on the challenging Celeb-DF dataset.
* **Computationally Efficient**: Requires only **~1.8B FLOPs** for inference, a significant reduction from the 8.4B FLOPs of the original Xception-based model.
* **Interactive Web Interface**: A simple and responsive web app to upload a face image and receive an instant **Real** or **Fake** prediction with a confidence score.

---

## 🛠️ Installation and Setup

### 1. Prerequisites

* Python 3.8 or higher
* Git
* NVIDIA GPU (Recommended for faster inference)

### 2. Clone the Repository

```bash
# PASTE YOUR GITHUB REPO URL HERE
git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
cd [your-repo-name]
