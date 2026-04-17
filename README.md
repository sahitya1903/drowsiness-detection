# Driver Drowsiness Detection

Driver drowsiness detection project built with TensorFlow/Keras, OpenCV, and Haar Cascade classifiers. The repository contains the training workflow, a saved model, and a notebook for webcam-based eye-state monitoring.

## Overview

This project classifies eye state as `Open_Eyes` or `Closed_Eyes` and uses that prediction as the basis for a drowsiness alert pipeline. The current repository is organized around notebooks:

- `Model_train.ipynb` for dataset preparation, model training, evaluation, and sample-image testing
- `main.ipynb` for loading the trained model and running webcam inference with face and eye detection

## Features

- MobileNet-based transfer learning model
- Binary eye-state classification for drowsiness detection
- Webcam inference workflow using OpenCV
- Haar Cascade face and eye detection
- Saved trained model included in the repository
- Sample test images for quick validation

## Project Directory

```text
drowsiness-detection/
├── README.md
├── LICENSE
├── .gitignore
├── Train_Dataset/
│   ├── Closed_Eyes/
│   └── Open_Eyes/
├── Test_Dataset/
│   ├── Closed_Eyes/
│   └── Open_Eyes/
├── Model_train.ipynb
├── main.ipynb
├── model.keras
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
├── test1.png
├── test2.png
└── test_img.jpg/
```

## File Guide

| Path | Purpose |
| --- | --- |
| `Model_train.ipynb` | Trains the eye-state classification model and saves it as `model.keras` |
| `main.ipynb` | Loads the trained model and runs live webcam-based detection |
| `model.keras` | Saved trained Keras model used during inference |
| `haarcascade_frontalface_default.xml` | Face detection cascade used by OpenCV |
| `haarcascade_eye.xml` | Eye detection cascade used by OpenCV |
| `test1.png`, `test2.png`, `test_img.jpg` | Sample images for testing and demonstration |
| `Train_Dataset/` | Training dataset directory with `Closed_Eyes` and `Open_Eyes` subfolders |
| `Test_Dataset/` | Test dataset directory with `Closed_Eyes` and `Open_Eyes` subfolders |

## Dataset Structure

Both dataset folders are expected to follow this class-based layout:

```text
Train_Dataset/
├── Closed_Eyes/
└── Open_Eyes/

Test_Dataset/
├── Closed_Eyes/
└── Open_Eyes/
```

Images are loaded directly from these folders in the training notebook.

## Model Workflow

1. Prepare training and test images inside the dataset folders.
2. Open `Model_train.ipynb`.
3. Train the MobileNet-based classifier.
4. Save the trained model as `model.keras`.
5. Open `main.ipynb` to run webcam inference using the saved model.

## How To Run

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

This repository does not currently include a `requirements.txt`, so install the core packages manually:

```bash
pip install tensorflow opencv-python numpy matplotlib jupyter
```

### 3. Train the model

Run `Model_train.ipynb` from Jupyter Notebook or VS Code Notebook support.

### 4. Run webcam detection

Open `main.ipynb`, ensure `model.keras` is present, then run the notebook cells to start webcam-based monitoring.


## Future Improvements

- Convert notebook workflows into Python scripts
- Add an audible alert module
- Add a `requirements.txt`
- Add evaluation metrics and result snapshots
- Package the webcam pipeline into a single runnable app

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
