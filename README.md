# Driver Drowsiness Detection and Alert System

A deep learning–based system that detects driver drowsiness in real time and triggers an alert to help prevent fatigue-related accidents.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Sample Testing](#sample-testing)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Work in Progress](#work-in-progress)
- [License](#license)

---

## Overview

Driver fatigue is one of the leading causes of road accidents worldwide. This project builds an intelligent drowsiness detection system that monitors a driver's eye state (open/closed) and raises an alert whenever signs of drowsiness are detected.

The system leverages **transfer learning** on the lightweight **MobileNet** architecture, making it efficient enough to run on resource-constrained hardware while still achieving high accuracy.

---

## Features

- **Transfer Learning** – Fine-tuned MobileNet pretrained on ImageNet for fast convergence and high accuracy.
- **High Accuracy** – ~97% validation accuracy on a held-out test set.
- **Large Training Dataset** – Trained on a curated dataset of 65,000+ images covering open/closed eye states under various lighting and head-pose conditions.
- **Static Image Testing** – Supports inference on arbitrary (unknown) images to verify model generalisation.
- **Alert System** – Triggers an audible/visual alert when drowsiness is detected.
- **Real-Time Monitoring** *(work in progress)* – Integration with webcam feed for continuous driver monitoring.

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Base model | MobileNet (pretrained on ImageNet) |
| Strategy | Transfer learning – base layers frozen, custom classification head added |
| Input size | 224 × 224 × 3 |
| Output classes | 2 (Drowsy / Alert) |
| Optimizer | Adam |
| Loss function | Categorical cross-entropy |

The lightweight MobileNet backbone ensures low latency, which is critical for real-time inference.

---

## Dataset

- **Total images**: ~65,000
- **Classes**: Drowsy (eyes closed / head drooping) and Alert (eyes open)
- **Sources**: Publicly available eye-state and driver-monitoring datasets
- **Preprocessing**: Resized to 224 × 224, normalised pixel values to [0, 1], data augmentation (horizontal flip, brightness adjustment, zoom)

---

## Model Performance

| Metric | Value |
|--------|-------|
| Training accuracy | ~98% |
| Validation accuracy | ~97% |

The model generalises well to unseen images, as confirmed by sample testing with independent images not present in the training or validation sets.

---

## Sample Testing

Inference on unknown images can be run independently of the real-time pipeline:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('drowsiness_model.h5')

img = image.load_img('path/to/test_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
label = 'Drowsy' if np.argmax(prediction) == 0 else 'Alert'
print(f'Prediction: {label}')
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/sahitya1903/drowsiness-detection.git
cd drowsiness-detection

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** A `requirements.txt` will be added once the real-time integration module is finalised.

---

## Usage

### Train the Model

```bash
python train.py
```

### Evaluate on Validation Set

```bash
python evaluate.py
```

### Run Inference on a Single Image

```bash
python predict.py --image path/to/image.jpg
```

### Real-Time Detection *(coming soon)*

```bash
python detect_realtime.py
```

---

## Project Structure

```
drowsiness-detection/
├── dataset/                # Training and validation images
├── models/                 # Saved model weights (.h5 / SavedModel)
├── notebooks/              # Jupyter notebooks for experimentation
├── train.py                # Model training script
├── evaluate.py             # Model evaluation script
├── predict.py              # Single-image inference script
├── detect_realtime.py      # Real-time webcam detection (WIP)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Work in Progress

- [ ] Real-time webcam integration using OpenCV
- [ ] Audible alert system (buzzer / audio notification)
- [ ] Face and eye detection preprocessing pipeline (Haar Cascade / Dlib)
- [ ] Deployment on edge devices (Raspberry Pi / Jetson Nano)
- [ ] REST API for remote monitoring

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.