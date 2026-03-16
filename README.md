🕵️‍♂️ Spatiotemporal Deepfake Detection Pipeline
A production-ready Deep Learning pipeline deployed on Hugging Face, engineered to detect manipulated video content. This architecture moves beyond standard frame-by-frame spatial classification by implementing a Temporal Attention Mechanism over a Bidirectional LSTM, allowing the network to identify microscopic, unnatural facial movements and blending boundaries over time.

🚀 Live Demo
Try the model in your browser: Launch Hugging Face Space

🧠 System Architecture
The pipeline processes .mp4 files through a custom extraction protocol and a hybrid feature-extraction network:

Smart Frame Extraction (OpenCV): Dynamically samples a sequence of 15 equidistant frames across the video duration, utilizing center-cropping to isolate primary subjects while minimizing VRAM overhead.

Spatial Backbone (EfficientNet-B0): Acts as the primary feature extractor.

Engineering Note: Instead of using fully frozen ImageNet weights, the top blocks (7 and above) are unfrozen during training. This allows the CNN to learn dataset-specific deepfake artifacts (like GAN-generated jawline blurring) while retaining generalized edge detection in the lower layers.

Temporal Processing (Bi-LSTM): A 2-layer Bidirectional LSTM (hidden size: 256) analyzes the temporal sequence of the 15 extracted spatial feature vectors to detect jitter and unnatural micro-expressions.

Attention Mechanism: Dynamically assigns weighted importance to specific frames in the sequence, forcing the classifier to focus heavily on the exact frames where deepfake glitches manifest.

🔬 Training Dynamics & Generalization
Training deepfake detectors on static datasets often leads to severe Domain Shift (the model memorizes dataset-specific video compression rather than actual facial manipulation). To combat this, the V2.1 training loop introduces strict regularization:

Stochastic Noise Injection: A custom torch.randn_like layer injects 2% digital static over the tensors during the forward pass. This aggressively destroys standard dataset compression artifacts, forcing the EfficientNet blocks to hunt for structural deepfake flaws.

Dynamic Learning Rate: Utilizes ReduceLROnPlateau (factor=0.5, patience=2) to stabilize the partial CNN unfreezing, ensuring the loss landscape converges smoothly without catastrophic forgetting of the pre-trained ImageNet weights.

Spatial Augmentation: 50% randomized horizontal tensor flipping to prevent background and lighting memorization.

💻 Local Setup & Inference
1. Clone the repository & Install dependencies

Bash
git clone https://github.com/sarvjais12/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
2. Download Weights
Pull the pre-trained weights (.pth file) from the Hugging Face repository and place them in the root directory.

3. Launch the Gradio Server

Bash
python app.py
🗺️ Roadmap & Future Iterations
[x] Initial PyTorch training on Kaggle Deep Fake Detection subset.

[x] Integrate Bi-LSTM + Attention architecture.

[x] Deploy interactive inference API via Hugging Face Spaces.

[ ] V3.0: Fine-tune the current architecture on the Celeb-DF dataset to improve zero-shot generalization against high-fidelity, Hollywood-grade deepfakes (e.g., the Morgan Freeman deepfake).

Built and deployed by sarvjais12.
