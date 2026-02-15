from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import io

# 1. Re-define the Architecture (Must match your training script exactly)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Setup App and Load Model
app = FastAPI(title="CIFAR-10 Image Classifier")

device = torch.device("cpu") # Standard for basic API deployment
model = SimpleCNN()
model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
model.eval()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3. Define the Inference Transform
# This must match the normalization used during training!
inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Pre-process the image
    input_tensor = inference_transform(image).unsqueeze(0) # Add batch dimension
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        # Apply Softmax to get probabilities (The Calculus part!)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, predicted = torch.max(probabilities, 0)
    
    return {
        "prediction": classes[predicted.item()],
        "confidence": float(conf),
        "status": "success"
    }