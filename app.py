import gradio as gr
import torch
import cv2
import numpy as np
from torchvision import models
import torch.nn as nn

# --- ARCHITECTURE MUST MATCH EXACTLY ---
class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.feature_extractor = base.features 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(1280, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Linear(512, 1)
        
    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        f = self.pool(self.feature_extractor(x)).view(b, s, 1280)
        lstm_out, _ = self.lstm(f)
        att_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(att_weights * lstm_out, dim=1)
        return self.classifier(context)

class ProFrameExtractor:
    def __init__(self, size=224): self.size = size
    def get_frames(self, path, num_frames=45):
        cap = cv2.VideoCapture(path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0: return None
        step = max(1, total // num_frames)
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret: break
            h, w, _ = frame.shape
            min_dim = min(h, w)
            start_h, start_w = (h - min_dim) // 2, (w - min_dim) // 2
            frame = frame[start_h:start_h+min_dim, start_w:start_w+min_dim]
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.size, self.size))
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        while len(frames) < num_frames: frames.append(np.zeros((self.size, self.size, 3), dtype=np.float32))
        return np.array(frames)

# --- INITIALIZATION ---
# Update this string to match the file you upload to the Space
MODEL_PATH = "deepfake_model_e10.pth" 
device = torch.device('cpu') # Hugging Face free tier uses CPU

model = DeepfakeModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- PREDICTION LOGIC ---
def predict_video(video_path):
    if video_path is None:
        return "Please upload a video."
        
    extractor = ProFrameExtractor()
    frames = extractor.get_frames(video_path)
    if frames is None:
        return "Error reading video file."
        
    tensor_frames = torch.tensor(frames.tolist()).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_frames)
        probability = torch.sigmoid(output).item()
        
    prediction = "FAKE (Manipulated)" if probability > 0.5 else "REAL (Authentic)"
    confidence = probability if prediction.startswith("FAKE") else 1 - probability
    
    return f"Prediction: {prediction}\nConfidence: {confidence*100:.2f}%"

# --- GRADIO UI ---
demo = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload Video to Test"),
    outputs=gr.Textbox(label="Analysis Result"),
    title="Deepfake Video Detection",
    description="Spatial-temporal deepfake detection using EfficientNet-B0 and Bi-LSTM with Attention. Upload an .mp4 to analyze.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()