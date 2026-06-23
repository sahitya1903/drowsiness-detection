# AlertDrive

A real-time driver drowsiness detection system built with **TensorFlow/Keras (MobileNet Transfer Learning)**, **OpenCV**, and **Streamlit**. The project provides both a local Jupyter notebook development environment and a browser-based web application utilizing WebRTC for real-time video streaming and inference.

---

## 🌟 Key Features

- **Transfer Learning Model**: Employs a pre-trained **MobileNet** architecture fine-tuned for high-accuracy binary classification (Open vs. Closed eyes).
- **Dual Inference Workflows**:
  - 🖥️ **Web Application**: Streamlit-based interface (`webdemo.py`) with `streamlit-webrtc` for real-time, low-latency browser-based inference.
  - 📓 **Jupyter Notebook**: OpenCV-based webcam window-based monitoring (`main.ipynb`) for local prototyping.
- **Drowsiness Alert System**: Tracks eye closure duration. If eyes are closed for more than **2 seconds**, a visual "SLEEP ALERT!" warning is triggered.
- **Multi-face and Eye Cascades**: Integrates OpenCV Haar Cascades for fast face and eye region-of-interest (ROI) detection.
- **Embedded Model**: The trained model (`model.keras`) is bundled directly in the workspace, ready for immediate inference.

---

## 📂 Project Directory Structure

```text
alert-drive/
├── .devcontainer/                      # Dev container settings
├── .gitignore                          # Standard git exclusions
├── LICENSE                             # MIT License
├── README.md                           # Project documentation
├── Dataset.zip                         # [Git Ignored] Compressed training & validation dataset
├── Model_train.ipynb                   # Dataset preparation, training, and evaluation notebook
├── main.ipynb                          # Local OpenCV-based webcam inference notebook
├── webdemo.py                          # Streamlit WebRTC application for browser-based inference
├── model.keras                         # Trained Keras classification model
├── requirements.txt                    # Project package dependencies
├── haarcascade_frontalface_default.xml # Face detection Haar cascade
├── haarcascade_eye.xml                 # Eye detection Haar cascade
├── test1.png                           # Sample eye-state image (Open)
├── test2.png                           # Sample eye-state image (Closed)
└── test_img.jpg                        # Sample detection test image
```

---

## 📊 Dataset & Directory Structure

> [!IMPORTANT]
> **Git Size Limit & Ignored File**:
> Due to file size limits, **`Dataset.zip`** (approx. 336 MB) is excluded from version control via `.gitignore`. 
> 
> To train the model yourself:
> 1. Acquire `Dataset.zip` (containing the labeled training images).
> 2. Place `Dataset.zip` directly in the project root directory.
> 3. Extract the zip file to create the `Train_Dataset/` and `Test_Dataset/` folders.

Once extracted, the directories should follow this class-based structure:

```text
Train_Dataset/
├── Closed_Eyes/
└── Open_Eyes/

Test_Dataset/
├── Closed_Eyes/
└── Open_Eyes/
```

---

## 🗺️ File Guide

| File / Folder | Purpose |
| --- | --- |
| `webdemo.py` | Runs the Streamlit web application with client-side WebRTC camera feeds and server-side model inference. |
| `Model_train.ipynb` | Handles dataset extraction, pre-processing, transfer learning model training, and saves the weights to `model.keras`. |
| `main.ipynb` | Loads the trained model and performs local webcam inference in an OpenCV pop-up window. |
| `model.keras` | Pre-trained MobileNet Keras model weights. |
| `requirements.txt` | Defines core libraries needed to run, train, and serve the application. |
| `Dataset.zip` | Archive containing the raw eye-state dataset folder layout. |
| `haarcascade_*.xml` | Pre-trained Haar Cascade XMLs for face and eye region localization. |

---

## ⚡ How to Set Up and Run

Follow these instructions to set up the environment and run either the web app or the notebooks.

### 1. Create a Virtual Environment

Open your terminal in the project directory and run:

```bash
# Create environment
python -m venv .venv

# Activate environment
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

Use the `requirements.txt` file to install all the necessary packages:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> `requirements.txt` uses `opencv-python-headless` which is suitable for headless and web server environments. If you want to run `main.ipynb` with local OpenCV GUI windows (e.g., `cv2.imshow()`), you should install GUI-enabled OpenCV:
> ```bash
> pip uninstall opencv-python-headless
> pip install opencv-python
> ```

---

## 🚀 Running the Applications

### Option A: Launch the Web App (Recommended)

To run the modern, browser-based WebRTC interface:

```bash
streamlit run webdemo.py
```

- This will automatically open `http://localhost:8501` in your browser.
- Click **Start** under the video streamer to allow camera permissions.
- The system will detect your face and eyes, run real-time inference, and highlight eyes with bounding boxes (Green for Open, Red for Closed).
- Keeping your eyes closed for more than 2 seconds will display a bright red **SLEEP ALERT!** banner on the stream.

### Option B: Local Notebook Inference

If you want to debug or view local OpenCV windows:
1. Open `main.ipynb` in VS Code or Jupyter Notebook.
2. Select your virtual environment kernel (`.venv`).
3. Run the cells sequentially to trigger a local camera preview window.
4. Press `q` in the OpenCV window to exit.

### Option C: Re-training the Model

If you wish to re-train the model or fine-tune with new datasets:
1. Make sure you have downloaded and placed `Dataset.zip` in the root workspace directory (see [Dataset & Directory Structure](#-dataset--directory-structure)).
2. Unzip the dataset files:
   ```bash
   # Extract Dataset.zip into the root directory
   # On Windows (PowerShell):
   Expand-Archive -Path Dataset.zip -DestinationPath .
   ```
3. Verify that `Train_Dataset` and `Test_Dataset` are created in the project root.
4. Open `Model_train.ipynb` and execute cells to train the MobileNet transfer learning layers.
5. The notebook will automatically output `model.keras` upon training completion.

---

## 📈 Future Scope & Improvements

Although the project is currently stable and complete, future versions could implement:
- **HTML5 Web Audio API Alert**: Triggering a loud audible alarm buzzer in the user's browser when a sleep alert is flagged.
- **Dockerization**: Containerizing the Streamlit application for unified multi-platform deployment.
- **Fatigue Indicators**: Integrating other fatigue metrics such as yawning rate or head tilt angle.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
