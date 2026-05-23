# Chest X-Ray Pneumonia Classification using Transfer Learning

This repository contains a collection of deep learning pipelines developed for automatic pneumonia detection from chest X-ray images using Transfer Learning and Convolutional Neural Networks (CNNs). The project focuses on evaluating and optimizing state-of-the-art architectures for medical image classification using TensorFlow and Keras.

The implemented pipelines were developed as part of a research study in Computer Vision and Medical Imaging, emphasizing reproducibility, modularity, and experimental comparison between different architectures and optimization strategies.

---

# Implemented Architectures

The repository currently includes pipelines based on:

- ResNet50
- DenseNet121
- EfficientNetB0
- Mamba-based architectures (experimental)

All models use pre-trained ImageNet weights and fine-tuning strategies for transfer learning.

---

# Main Features

The pipelines include several optimization and regularization techniques commonly used in modern deep learning workflows:

- Transfer Learning
- Fine-Tuning
- Dynamic Data Augmentation
- Focal Loss
- Class Balancing
- Cosine Decay Learning Rate Scheduler
- Dropout Regularization
- L2 Regularization
- Test-Time Augmentation (TTA)
- Threshold Optimization
- Early Stopping
- Model Checkpointing
- ROC Curve and AUC Analysis

---

# Dataset

The experiments were conducted using the public Chest X-Ray Pneumonia dataset available on Kaggle.

Classes:
- Pneumonia
- Normal

All images are resized to:

```python
224 × 224 × 3
```

---

# Project Structure

```bash
.
├── pipelines/
│   ├── resnet50/
│   ├── densenet121/
│   ├── efficientnetb0/
│   └── mamba/
│
├── figures/
│
├── histories/
│
├── models/
│
├── results/
│
└── README.md
```

---

# Training Features

The training pipelines support:

- Saving trained models
- Saving training histories (.csv and .json)
- Automatic plot generation
- Confusion Matrix generation
- ROC Curve generation
- Threshold optimization for best F1-score
- GPU acceleration with Google Colab

---

# Metrics Evaluated

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

# Results

Among the evaluated architectures, ResNet50 achieved the best overall performance, obtaining:

- Accuracy: 94%
- Macro F1-Score: 94%
- ROC-AUC: 98.5%

The results demonstrate that transfer learning combined with modern optimization strategies can provide highly competitive performance for pneumonia detection using chest radiographs.

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- OpenCV
- Google Colab

---

# Future Work

Future improvements may include:

- Vision Transformers (ViT)
- Ensemble Learning
- Explainable AI (Grad-CAM)
- Federated Learning
- Lightweight mobile architectures
- Self-supervised learning approaches

---

# Citation

If you use this repository in your research, please cite the corresponding paper/project.

---

# Author

Dante Veloso

M.Sc. Research in Computer Vision and Medical Imaging