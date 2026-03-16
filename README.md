
# 🕵️‍♂️ Spatiotemporal Deepfake Engine (PyTorch + Bi-LSTM)

### **Advanced Video Manipulation Detection with Temporal Attention**

This project is an advanced deep learning implementation developed to solve the limitations of standard, frame-by-frame deepfake classifiers. It uses a hybrid **CNN + RNN** architecture to analyze not just spatial artifacts (like blurry jawlines), but the **temporal inconsistencies** (unnatural micro-expressions) across a video sequence.

---

## 🚀 Key Features

* **Domain-Shift Resilience (V2.1)**: Solved the dataset memorization problem by injecting 2% digital static noise and using 50% randomized horizontal flipping during training.
* **Partial Backbone Unfreezing**: Kept the base ImageNet weights frozen for edge detection, but unfroze the top EfficientNet blocks so the CNN could actively learn GAN-generated visual artifacts.
* **Smart Frame Extraction**: Uses OpenCV to intelligently sample 15 equidistant frames across the video, automatically center-cropping around the subject to optimize VRAM.
* **Temporal Attention Mechanism**: Dynamically assigns weights to specific frames, forcing the model to focus on the exact moments a deepfake glitch occurs.
* **Dynamic Learning Rate**: Integrated PyTorch's `ReduceLROnPlateau` scheduler to optimize convergence without catastrophic forgetting.

---

## 🏗️ How it Works (The Architecture)

The system processes video files through a multi-stage spatiotemporal pipeline:

1. **Spatial Feature Extraction**: The unfrozen `EfficientNet-B0` extracts 1280-dimensional feature vectors from each of the 15 cropped frames.
2. **Temporal Sequence Analysis**: A 2-layer **Bidirectional LSTM** (hidden size: 256) processes the chronological sequence to detect unnatural movement and jitter over time.
3. **Attention Weighting**: A custom attention layer scores the LSTM outputs, highlighting the specific frames with the highest probability of manipulation.
4. **Binary Classification**: A fully connected layer outputs the final prediction (Real vs. Fake) and passes it to the UI.

---

## 🛠️ Tech Stack

* **Logic**: PyTorch, Torchvision
* **Spatial Backbone**: EfficientNet-B0
* **Temporal Network**: Bi-LSTM + Attention layer
* **Video Processing**: OpenCV, NumPy
* **Interface**: Gradio (via Hugging Face Spaces) :https://huggingface.co/spaces/Sarvjais12/Deepfake_Video_Detection

---

## 📊 Sample Use Case: The Generalization Gap

When tested against different tiers of synthetic media:

* **Standard Dataset Fake**: "01_03__outside_talking.mp4" -> Triggers **Spatial Detection** (catches automated blending boundaries).
* **High-Fidelity Fake**: "Morgan Freeman Deepfake" -> Bypasses spatial checks due to Hollywood-grade smoothing, proving the necessity of the **Temporal Attention** V2.1 upgrades to catch unnatural sequential movement.

---

## 🔗 Live Demo

Try it out on Hugging Face Spaces: [Deepfake Video Detection](https://huggingface.co/spaces/Sarvjais12/Deepfake_Video_Detection)

---
