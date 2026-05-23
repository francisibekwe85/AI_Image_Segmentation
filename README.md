# AI Image Segmentation

Medical AI segmentation and classification experiments using advanced deep learning architectures for medical image analysis.

## 📋 Overview

This repository contains comprehensive implementations of medical image segmentation and classification tasks using state-of-the-art deep learning models. The project focuses on breast cancer detection, prostate cancer segmentation, tumor segmentation, and segmentation-guided classification using various datasets and architectures.

## 🏥 Medical Applications

- **Breast Cancer Segmentation** - Ultrasound breast lesion segmentation using the BUSI dataset
- **Prostate Cancer Segmentation** - Multi-modal prostate MRI segmentation using nnU-Net
- **Tumor Segmentation** - Brain tumor segmentation from multi-modal MRI data
- **Segmentation-Guided Classification** - Combining segmentation masks for improved tumor classification

## 📁 Repository Structure

```
AI_Image_Segmentation/
├── Breast_Cancer_Seg.ipynb              # BUSI ultrasound breast lesion segmentation
├── Prostate_nnU-Net_Segmentation.ipynb  # nnU-Net for multi-modal prostate segmentation
├── Roboflow_Tumor_Segmentation.ipynb    # Brain tumor MRI segmentation
├── Seg_Guided_Tumor_Classification.ipynb # Classification using segmentation masks
└── README.md                            # This file
```

## 🔧 Technologies & Frameworks

- **Deep Learning:** PyTorch, TensorFlow/Keras
- **Medical Imaging:** SimpleITK, nibabel, MONAI
- **Segmentation Models:** 
  - U-Net (standard convolutional segmentation)
  - Attention U-Net (attention mechanisms)
  - nnU-Net (self-configuring architecture)
- **Data Processing:** OpenCV, scikit-image, Albumentations
- **Analysis & Visualization:** NumPy, Pandas, Matplotlib, Seaborn
- **Radiomics:** Radiomic feature extraction and analysis

## 📊 Notebooks

### 1. **Breast_Cancer_Seg.ipynb**
- **Dataset:** BUSI (Breast Ultrasound Images)
- **Architecture:** U-Net / Attention U-Net
- **Task:** Binary segmentation of benign/malignant lesions
- **Features:** Data augmentation, model training, evaluation metrics (Dice, IoU)

### 2. **Prostate_nnU-Net_Segmentation.ipynb**
- **Dataset:** Multi-modal prostate MRI
- **Architecture:** nnU-Net (self-configuring)
- **Task:** Prostate gland and zone segmentation
- **Features:** Automatic hyperparameter optimization, 3D segmentation

### 3. **Roboflow_Tumor_Segmentation.ipynb**
- **Dataset:** Brain tumor segmentation challenge (via Roboflow)
- **Architecture:** U-Net variants
- **Task:** Multi-class brain tumor segmentation (Necrosis, Edema, Enhancing Tumor)
- **Features:** Multi-modal MRI processing, comprehensive evaluation

### 4. **Seg_Guided_Tumor_Classification.ipynb**
- **Approach:** Two-stage pipeline (Segmentation → Classification)
- **Architecture:** Segmentation masks guide classification network
- **Task:** Improve tumor classification using spatial information
- **Features:** Feature engineering from segmentation masks, ensemble methods

## 📦 Datasets

The project uses publicly available medical imaging datasets:

- **BUSI Dataset** - Breast Ultrasound Images with segmentation masks
- **Prostate MRI Data** - Multi-modal 3D prostate imaging
- **Brain Tumor Dataset** - Multi-modal MRI for tumor segmentation
- **Roboflow Datasets** - Curated medical imaging collections

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended for faster training)
- CUDA 11.0+ (if using GPU)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/francisibekwe85/AI_Image_Segmentation.git
cd AI_Image_Segmentation
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Key Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
tensorflow>=2.6.0
monai>=0.6.0
SimpleITK>=2.1.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
scikit-image>=0.18.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
jupyter>=1.0.0
```

## 🚀 Usage

### Running Jupyter Notebooks

1. **Launch Jupyter:**
```bash
jupyter notebook
```

2. **Open and run a notebook:**
   - Select `Breast_Cancer_Seg.ipynb` or other notebooks
   - Run cells sequentially (Shift + Enter)
   - Follow markdown instructions for dataset preparation

### Training a Segmentation Model

Typical workflow in each notebook:

1. **Data Preparation:** Load and preprocess medical images
2. **Data Augmentation:** Apply transformations (rotation, flip, intensity shifts)
3. **Model Architecture:** Define U-Net or nnU-Net
4. **Training:** Optimize using Adam/SGD with appropriate loss functions
5. **Evaluation:** Compute Dice score, Intersection over Union (IoU), Hausdorff Distance
6. **Visualization:** Display predictions vs. ground truth masks

### Example Code Snippet

```python
import torch
from monai.networks.nets import UNet

# Define model
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64),
    strides=(2, 2)
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.DiceLoss()

for epoch in range(100):
    # Forward pass
    output = model(images)
    loss = loss_fn(output, masks)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 📈 Model Evaluation Metrics

- **Dice Coefficient (F1 Score):** Overlap between predicted and ground truth
- **Intersection over Union (IoU):** Ratio of intersection to union
- **Sensitivity/Recall:** Ability to identify positive pixels
- **Specificity:** Ability to identify negative pixels
- **Hausdorff Distance:** Maximum boundary distance
- **Radiomics:** Shape, texture, and intensity features

## 🔬 Key Findings

- **Attention mechanisms** improve segmentation accuracy on small lesions
- **nnU-Net's** automatic configuration achieves state-of-the-art results with minimal tuning
- **Segmentation-guided classification** outperforms classification-only approaches
- **3D models** better capture spatial context for volumetric medical data
- **Data augmentation** is critical for small medical imaging datasets

## 🎯 Applications

- **Clinical Decision Support:** Assist radiologists in tumor detection/segmentation
- **Radiomics Analysis:** Extract quantitative imaging biomarkers
- **Treatment Planning:** Define regions of interest for radiation therapy
- **Research:** Benchmark deep learning architectures on medical data
- **Quality Control:** Automated segmentation for consistency

## 📚 Learning Resources

- [MONAI Documentation](https://docs.monai.io/)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Attention U-Net Paper](https://arxiv.org/abs/1804.03999)
- [nnU-Net Paper](https://arxiv.org/abs/1904.08128)
- [Medical Image Analysis with Deep Learning](https://www.deeplearningbook.org/)

## 💡 Future Improvements

- [ ] Add 3D segmentation models (3D U-Net variants)
- [ ] Implement semi-supervised learning for limited labels
- [ ] Add uncertainty quantification
- [ ] Develop web interface for inference
- [ ] Integrate with DICOM viewers (Cornerstone.js)
- [ ] Add model interpretability (GradCAM, attention maps)
- [ ] Benchmark against state-of-the-art models

## ⚠️ Disclaimer

These implementations are for **educational and research purposes only**. They are not intended for clinical use without proper validation, regulatory approval, and clinical oversight by qualified medical professionals.

## 📝 License

This project is open source. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share additional medical imaging datasets

## 📧 Contact

For questions or collaboration inquiries, please reach out via GitHub.

## 🙏 Acknowledgments

- BUSI Dataset Authors
- Roboflow for curated medical datasets
- MONAI community for excellent medical imaging tools
- PyTorch and TensorFlow developers

---

**Last Updated:** May 2026  
**Repository:** [francisibekwe85/AI_Image_Segmentation](https://github.com/francisibekwe85/AI_Image_Segmentation)
