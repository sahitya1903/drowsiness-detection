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
| Training accuracy | ~96% |
| Validation accuracy | ~97% |

The model generalises well to unseen images, as confirmed by sample testing with independent images not present in the training or validation sets.

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


## Work in Progress

- [ ] Real-time webcam integration using OpenCV
- [ ] Audible alert system (buzzer / audio notification)
- [ ] Face and eye detection preprocessing pipeline (Haar Cascade / Dlib)
- [ ] Deployment on edge devices (Raspberry Pi / Jetson Nano)
- [ ] REST API for remote monitoring

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
