# Handwriting_Recodnization_Using_CNN
## 📋 Project Overview
This project implements a Convolutional Neural Network (CNN) for handwritten character recognition using the EMNIST dataset. The model achieves **88.36% test accuracy** on 47-class character classification (digits 0–9, uppercase A–Z, and balanced lowercase letters).

* **Tech Stack:** Python, PyTorch, Torchvision, EMNIST Dataset, CNN Architecture

---

## 🎯 Features
* **47-class character recognition** (digits + uppercase + lowercase letters)
* **Custom CNN architecture** with `BatchNorm2d` and `Dropout` for regularization
* **Data preprocessing pipeline** with normalization, tensor transformation, and orientation correction
* **Training and evaluation scripts** with detailed epoch-wise metrics
* **GPU-accelerated training** with automatic CUDA detection
* **Extensible design** for future CRNN-based word/sentence recognition

---

## 📊 Results

### Training Performance (5 Epochs)

| Epoch | Loss | Accuracy |
| :---: | :---: | :---: |
| **1** | 0.6729 | 78.08% |
| **2** | 0.4316 | 84.74% |
| **3** | 0.3834 | 86.15% |
| **4** | 0.3569 | 87.10% |
| **5** | 0.3381 | 87.66% |

**Final Test Accuracy:** **88.36%**
