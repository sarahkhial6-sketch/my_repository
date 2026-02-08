# Deepfake Video Detection Platform Using Multimodal Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

A robust, user-friendly web application designed to detect deepfake manipulation in videos and images. This platform utilizes **Vision Transformers (ViT)** for visual artifact detection and includes modules for audio spectrum analysis, providing a multimodal approach to media forensics.

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Future Scope](#future-scope)

## 🧐 About the Project

With the rise of Generative Adversarial Networks (GANs) and sophisticated AI generation tools, distinguishing between real and synthetic media has become a critical challenge.

This project addresses this by providing a **Deepfake Detection Platform** that:
1.  **Ingests Media:** Accepts user-uploaded videos or static images.
2.  **Extracts Frames:** Uses OpenCV to sample keyframes (1 frame/sec) for visual inspection.
3.  **AI Inference:** Runs frames through a pre-trained **Vision Transformer (ViT)** model (`dima806/deepfake_vs_real_image_detection`).
4.  **Audio Analysis:** visualizes audio frequency patterns (spectrograms) to assist in identifying anomalies.
5.  **Result Aggregation:** Computes a confidence score to classify media as **Real** or **Fake**.

## ✨ Key Features

* **Multimodal Input Support:** Seamlessly handles `.mp4`, `.avi`, `.jpg`, `.png`, and `.jpeg` files.
* **State-of-the-Art Model:** Leverages Hugging Face Transformers for high-accuracy artifact detection.
* **Intelligent Frame Extraction:** Balances performance and accuracy by analyzing temporal slices of video.
* **Audio Spectrograms:** Provides visual representation of audio data for forensic analysis.
* **Real-Time Feedback:** Built with Streamlit for instant analysis and results.
* **Smart Caching:** Utilizes `@st.cache_resource` to load heavy AI models only once, speeding up the user experience.

## 🏗 Technical Architecture

The system is modularized into three core components:

1.  **Frontend (Streamlit):**
    * Manages the user interface, file uploading, and result display.
2.  **Visual Extractor (OpenCV + Transformers):**
    * Decodes video streams.
    * Converts BGR frames to RGB.
    * Passes tensor data to the Vision Transformer pipeline.
3.  **Audio Feature Extractor:**
    * Extracts audio signals and generates spectrogram data for visualization.

## ⚙️ Installation

### Prerequisites
* Python 3.8 or higher
* pip (Python Package Manager)

### Step-by-Step Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/deepfake-detection-platform.git](https://github.com/your-username/deepfake-detection-platform.git)
    cd deepfake-detection-platform
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  **Run the Application**
    ```bash
    streamlit run main.py
    ```
    *(Note: Replace `main.py` with your actual Python filename if different)*

2.  **Analyze Media**
    * Open your browser to the local URL provided (usually `http://localhost:8501`).
    * Upload a video or image using the sidebar widget.
    * Click the **"Initialize Real-Time Detection"** button.
    * Review the frame-by-frame analysis and the final "Real vs Fake" probability score.

## 📦 Dependencies

The core libraries used in this project are:

* **`streamlit`**: Web application framework.
* **`transformers`**: For loading the pre-trained Deepfake Detection model.
* **`torch`**: The underlying deep learning framework.
* **`opencv-python`**: For video processing and frame extraction.
* **`pillow`**: For image manipulation.
* **`numpy`**: For numerical operations and audio data handling.

## 🔮 Future Scope

* **Advanced Audio Forensics:** Integration of MFCC (Mel-frequency cepstral coefficients) analysis to detect voice cloning.
* **Temporal Consistency Check:** Using RNNs/LSTMs to detect flickering artifacts across video frames.
* **Batch Processing:** Support for analyzing entire folders of media at once.
* **API Deployment:** Wrapping the detection logic in a FastAPI backend for mobile app integration.

## ⚠️ Disclaimer

This tool is intended for **educational and research purposes**. While the underlying model is effective, no AI detection tool is 100% accurate. Results should be used as an assistive indicator rather than absolute proof.

---
**Developed for the "Deepfake Video Detection Platform Using Multimodal Learning" Project.**
