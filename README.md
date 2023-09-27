# X-ray Transferable Polyrepresentation Learning

[![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-grey.svg?logo=python&logoColor=blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?logo=OpenCV&logoColor=white)](https://opencv.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PEP8](https://img.shields.io/badge/PEP8-compliant-brightgreen.svg)](https://www.python.org/dev/peps/pep-0008/)
[![DOI](https://img.shields.io/badge/DOI-in%20progress-b31b1b.svg)](https://arxiv.org/abs/TODO)


## Key Features

🔗 **Polyrepresentation Concept:** Explore the innovative concept of polyrepresentation, which seamlessly combines multiple representations from various sources to enhance machine learning performance.

🌐 **Transferability:** Discover how the created polyrepresentation can be effectively transferred to smaller datasets, demonstrating its potential as a practical and efficient solution for various image-related tasks.

🔬 **Multi-Faceted Approach:** Address classification problems comprehensively by leveraging multiple representations, including vector embeddings, self-supervised learning, and radiomic features.

📈 **Performance Boost:** Learn how polyrepresentation consistently outperforms single-model approaches, leading to improved results across different tasks and datasets.

🌍 **Versatility:** While initially applied to X-ray images, polyrepresentation's versatility makes it adaptable to diverse domains, offering a novel perspective on data representation.

## Installation

The package requirements are as follows:
```
pandas
numpy
matplotlib
scikit-learn
xgboost
joblib
dalex
seaborn
opencv-python
PyWavelets
torch
timm
neptune
albumentations
```

## Getting started

1. Clone the repository:
git clone https://github.com/Hryniewska/polyrepresentation.git
cd polyrepresentation

2. Install the required packages

3. For training Siamese Network, use: **Siamese_Network_training.ipynb**

4. For XGBoost training, use: **ML_training.ipynb**

5. For XGBoost inference, use: **ML_inference.ipynb**

6. Pretrained XGBoost model: **model_xgb_estimator.json**

## Reference

If you find our work useful, please cite our paper:

```
@article{Hryniewskapolyrepresentation,
	title={X-ray transferable polyrepresentation learning}, 
	author={Weronika Hryniewska and Przemys{\l}aw Biecek},
	journal = {preprint},
	year={2023}
}
