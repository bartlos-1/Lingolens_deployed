# LingoLens 👄🔍

LingoLens is a specialized computer vision and deep learning platform designed for automated lip-reading and visual speech recognition. By analyzing localized facial landmarks and mouth movements from video streams, the system maps spatiotemporal visual features to textual representations.

This repository handles the core pipeline architecture, optimized inference engine, and deployment assets for the model.

### BASED ON LIPNET RESEARCH PAPER BY UNIVERSITY OF OXFORD AND GOOGLE DEEPMIND
---

## 🌐 Live Deployments & Research
* **Interactive Web Application:** [Access the Live Demo](https://lingoread.streamlit.app/)
* **Research Paper:** Read the full architectural methodology: [Read the Paper](https://lingopaper.tiiny.site/)

---

## 🚀 Key Features
* **Spatiotemporal Feature Extraction:** Automated region-of-interest (ROI) isolation targeting lip contours utilizing OpenCV.
* **Deep Learning Architecture:** Utilizes a custom sequence-to-sequence network integrating Bidirectional Long Short-Term Memory (BiLSTM) layers to capture temporal alignment in continuous speech.
* **Optimized Pipeline:** Streamlined frame preprocessing and tokenization sequences designed for seamless model inference.

## 📊 Dataset & Architecture
The model was trained using the **GRID Corpus** dataset, a large-scale audio-visual sentence corpus. 
* **Input Pipeline:** Raw video format $\rightarrow$ Frame extraction $\rightarrow$ Face alignment & Mouth ROI isolation $\rightarrow$ Grayscale normalization.
* **Model Topology:** Spatiotemporal Convolutions coupled with Bidirectional LSTMs to process sequential visual frames and decode text sequences.

## ⚙️ Tech Stack & Dependencies
* **Core Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV (cv2)
* **Data Processing:** NumPy, Matplotlib

*All environment dependencies are fully managed and detailed in the root `requirements.txt` file.*

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/bartlos-1/Lingolens_deployed.git](https://github.com/bartlos-1/Lingolens_deployed.git)
   cd Lingolens_deployed
