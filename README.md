<div align="center">

🔴 Spatiotemporal Deepfake Detection Engine ⬛
An advanced, production-ready Deep Learning pipeline engineered to detect manipulated video content by analyzing spatial artifacts and temporal inconsistencies.

</div>

🩸 The Engineering Journey: From V1 to V2.1
Building a robust deepfake detector is an ongoing battle against Domain Shift. This repository documents the evolution of the model architecture and the training pipeline to handle increasingly complex synthetic media.

Phase 1: The Baseline (V1)
The initial model utilized a strictly frozen ImageNet backbone (EfficientNet-B0) paired with a Bi-LSTM. While it achieved an impressive 0.38 validation loss, it suffered from severe overfitting to the dataset's specific video compression algorithms. It could catch standard manipulations but was blind to high-fidelity, Hollywood-grade deepfakes (like the infamous Morgan Freeman deepfake) where spatial glitches were manually smoothed out.

Phase 2: Domain-Shift Resilience & Smart Unfreezing (V2.1)
To force the model to learn universal deepfake mechanics (e.g., temporal jitter, blending boundaries) rather than memorizing dataset compression, the pipeline was overhauled with the following techniques:

Stochastic Digital Noise: Injecting a customized 2% torch.randn_like static noise layer during the forward pass to destroy easy spatial clues.

Partial Backbone Unfreezing: The top layers (Block 7+) of the EfficientNet were unfrozen, allowing the CNN to actively learn GAN-generated jawline blurring while the base retained edge detection.

Dynamic Learning Rate: Implemented a ReduceLROnPlateau scheduler to stabilize the unfreezing process and prevent catastrophic forgetting of the ImageNet weights.

🧠 Core Architecture
The system processes 15 equidistant frames through a hybrid CNN + RNN + Attention pipeline.

Python
import torch
import torch.nn as nn
from torchvision import models

class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial Feature Extractor
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_extractor = base.features 
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal Sequence Analyzer
        self.lstm = nn.LSTM(1280, 256, num_layers=2, batch_first=True, bidirectional=True)
        
        # Temporal Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128), 
            nn.Tanh(), 
            nn.Linear(128, 1)
        )
        
        # Binary Classifier
        self.classifier = nn.Linear(512, 1)
        
    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        
        # Extract features per frame
        f = self.pool(self.feature_extractor(x)).view(b, s, 1280)
        
        # Analyze temporal movement
        lstm_out, _ = self.lstm(f)
        
        # Apply attention to critical frames
        att_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(att_weights * lstm_out, dim=1)
        
        return self.classifier(context)
⚙️ Advanced Data Augmentation & Extraction
To maintain VRAM efficiency and feed the Bi-LSTM a dense temporal sequence, a custom ProFrameExtractor was built using OpenCV to intelligently sample and center-crop frames. During training, tensors are aggressively augmented.

Python
# V2.1 Augmentation Protocol Snippet
if self.is_train:
    # 50% Randomized Horizontal Mirroing
    if random.random() > 0.5: 
        frames_tensor = torch.flip(frames_tensor, dims=[3])
        
    # 2% Noise Injection to combat dataset memorization
    noise = torch.randn_like(frames_tensor) * 0.02
    frames_tensor = torch.clamp(frames_tensor + noise, 0, 1) 
🚀 Live Inference Deployment
The finalized v2_1_deepfake_model_e10.pth weights are currently serving inference in real-time via a custom Gradio API hosted on Hugging Face Spaces.

Test the Model: 🔴 Launch Deepfake Video Detection Space 

Local Setup
To run the inference server on your own hardware:

Bash
git clone https://github.com/sarvjais12/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
python app.py
<div align="center">

Engineered and trained by sarvjais12.

</div>
