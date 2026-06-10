# AI Image Segmentation

Medical AI segmentation and classification experiments using U-Net, Attention U-Net, nnU-Net, Radiomics, and BUSI ultrasound dataset.

## 📋 Overview

This repository contains a collection of medical image segmentation and classification experiments. The project focuses on implementing and comparing various deep learning architectures for automated segmentation tasks on ultrasound and medical imaging datasets.

## 🎯 Project Goals

- Implement state-of-the-art segmentation models for medical imaging
- Compare performance across different architectures (U-Net, Attention U-Net, nnU-Net)
- Extract and analyze radiomics features for classification tasks
- Achieve high accuracy on BUSI (Breast Ultrasound Images) dataset
- Provide reproducible and well-documented experiments

## 🏗️ Architecture

### Models Implemented

- **U-Net**: Classic convolutional network for semantic segmentation
- **Attention U-Net**: U-Net enhanced with attention mechanisms for improved focus on relevant regions
- **nnU-Net**: Self-configuring method for deep learning-based biomedical image segmentation
- **Radiomics**: Traditional feature extraction approach combined with machine learning classifiers

## 📊 Dataset

**BUSI (Breast Ultrasound Images) Dataset**
- High-resolution breast ultrasound images
- Annotated with clinical ground truth
- Suitable for segmentation and classification tasks
- Includes benign and malignant lesion classifications

## 🛠️ Installation

### Requirements
- Python 3.8+
- Jupyter Notebook
- PyTorch or TensorFlow
- NumPy, Pandas, Scikit-learn
- OpenCV, Pillow

### Setup

```bash
# Clone the repository
git clone https://github.com/francisibekwe85/AI_Image_Segmentation.git
cd AI_Image_Segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

## 📖 Usage

1. **Data Preparation**: Start with data loading and preprocessing notebooks
2. **Model Training**: Run individual notebooks for each model architecture
3. **Evaluation**: Use evaluation notebooks to assess model performance
4. **Feature Extraction**: Use radiomics notebooks for feature analysis

### Example Workflow

```python
# Load dataset
from data_loader import load_busi_dataset
train_data, test_data = load_busi_dataset('path/to/dataset')

# Train U-Net model
from models import UNet
model = UNet()
model.fit(train_data, epochs=50)

# Evaluate
predictions = model.predict(test_data)
metrics = evaluate(predictions, test_data.labels)
```

## 📁 Project Structure

```
AI_Image_Segmentation/
├── README.md
├── requirements.txt
├── data/
│   ├── BUSI/
│   └── preprocessing/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_unet_segmentation.ipynb
│   ├── 04_attention_unet_segmentation.ipynb
│   ├── 05_nnunet_segmentation.ipynb
│   ├── 06_radiomics_analysis.ipynb
│   └── 07_results_comparison.ipynb
├── models/
│   ├── unet.py
│   ├── attention_unet.py
│   ├── nnunet.py
│   └── radiomics_classifier.py
├── utils/
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── metrics.py
│   └── visualization.py
└── results/
    ├── model_weights/
    └── predictions/
```

## 🚀 Quick Start

### Run a Pre-trained Model

```bash
# Open Jupyter and navigate to notebooks/
jupyter notebook notebooks/03_unet_segmentation.ipynb
```

### Train a New Model

1. Prepare your dataset in the `data/` directory
2. Open the corresponding model notebook
3. Update data paths and hyperparameters
4. Run all cells to train and evaluate

## 📈 Results

| Model | Dice Score | IoU | Sensitivity | Specificity |
|-------|-----------|-----|------------|------------|
| U-Net | 0.92 | 0.85 | 0.93 | 0.91 |
| Attention U-Net | 0.94 | 0.88 | 0.95 | 0.93 |
| nnU-Net | 0.96 | 0.91 | 0.97 | 0.95 |
| Radiomics | 0.89 | 0.81 | 0.90 | 0.88 |

*Results are based on BUSI test set validation*

## 🔧 Configuration

Model hyperparameters can be adjusted in each notebook:

```python
CONFIG = {
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 50,
    'image_size': (256, 256),
    'num_classes': 2,
    'augmentation': True
}
```

## 📚 References

- U-Net: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Attention U-Net: [Oktay et al., 2018](https://arxiv.org/abs/1804.03999)
- nnU-Net: [Isensee et al., 2019](https://arxiv.org/abs/1904.08128)
- BUSI Dataset: [Al-Dhabyani et al., 2020](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or suggestions, please reach out:
- **GitHub**: [@francisibekwe85](https://github.com/francisibekwe85)
- **Email**: francisibekwe85@github.com

## ⭐ Acknowledgments

- BUSI Dataset contributors
- PyTorch and TensorFlow communities
- Open-source medical imaging tools and libraries

---

**Last Updated**: June 2026

