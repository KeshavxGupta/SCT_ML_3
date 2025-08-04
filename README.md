# Dogs vs Cats Image Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Table of Contents
- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Development](#development)
- [Future Roadmap](#future-roadmap)

## 🔍 Overview

An advanced machine learning implementation for binary image classification between dogs and cats, utilizing computer vision techniques and traditional machine learning approaches. The project demonstrates the effective use of Histogram of Oriented Gradients (HOG) for feature extraction combined with Support Vector Machine (SVM) classification.

### Key Features
- Robust image preprocessing pipeline
- Advanced feature extraction using HOG
- Optimized SVM classification with hyperparameter tuning
- Comprehensive evaluation metrics
- Parallel processing implementation

## 🏗 Technical Architecture

### Feature Extraction Pipeline
```
Raw Image → Grayscale → Resize (100x100) → Normalization → HOG Features → SVM Classification
```

### Technologies Used
- **Core**: Python 3.8+
- **Image Processing**: OpenCV
- **Machine Learning**: scikit-learn
- **Feature Extraction**: scikit-image
- **Data Processing**: NumPy
- **Visualization**: Matplotlib, Seaborn
- **Progress Tracking**: tqdm

## 💻 Requirements

```bash
numpy
opencv-python
scikit-image
scikit-learn
matplotlib
seaborn
tqdm
```

## Project Structure

```
.
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   ├── test/
│   │   ├── cats/
│   │   └── dogs/
│   └── validation/
│       ├── cats/
│       └── dogs/
├── organize_dataset.py
├── .py
├── .gitignore
└── README.md
```

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dogs-vs-cats
```

2. Create a virtual environment and activate it:
```bash
python -m venv myenv
# On Windows
myenv\Scripts\activate
# On Unix or MacOS
source myenv/bin/activate
```

3. Install the required packages:
```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib seaborn tqdm
```

4. Organize your dataset:
```bash
python organize_dataset.py
```

5. Run the classification:
```bash
python .py
```

## Features

- **Data Preprocessing**: 
  - Image resizing to 100x100 pixels
  - Grayscale conversion
  - Pixel normalization

- **Feature Extraction**:
  - HOG (Histogram of Oriented Gradients) features
  - Parallel processing for faster computation

- **Model Training**:
  - Support Vector Machine (SVM) classifier
  - GridSearchCV for hyperparameter optimization
  - Cross-validation for model evaluation

- **Evaluation Metrics**:
  - Accuracy score
  - Classification report
  - Confusion matrix visualization

## Model Details

The project uses an SVM classifier with the following hyperparameters tuned via GridSearchCV:
- C: [0.1, 1, 10]
- gamma: [0.0001, 0.001, 0.01]
- kernel: rbf

## Results

The model performance metrics include:
- Training accuracy
- Test accuracy
- Detailed classification report
- Visualized confusion matrix

## File Descriptions

- `organize_dataset.py`: Script to organize images into train/test/validation splits
- `.py`: Main classification script
- `.gitignore`: Specifies which files Git should ignore
- `README.md`: Project documentation

## Acknowledgments

- Dataset source: https://www.kaggle.com/competitions/dogs-vs-cats
- Any other acknowledgments or references

## Future Improvements

Potential areas for enhancement:
1. Implementation of deep learning models (CNN)
2. Data augmentation techniques
3. Model deployment capabilities
4. Real-time classification
5. Web interface for testing
