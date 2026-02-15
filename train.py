import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Device Configuration (Utilizing Apple Silicon/MPS for performance)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Deployment Device: {device}")

# 2. Data Engineering & Augmentation
# Normalization helps the Gradient Descent algorithm converge faster
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading standard CIFAR-10 benchmarks
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3. Convolutional Neural Network Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional Layers: Extracting spatial features (edges/textures)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Linear Layers: Classification based on extracted features
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8) # Flattening for the fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# 4. Optimizer & Loss Function (The Calculus Engine)
# CrossEntropyLoss requires the calculation of multi-class gradients
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
print("Starting Training Loop...")
epochs = 5
for epoch in range(epochs):
    model.train() # Enable training-specific layers (e.g. Dropout)
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients to prevent accumulation
        optimizer.zero_grad()
        
        # Forward Pass + Backward Pass (Backpropagation)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'Epoch {epoch+1}, Batch {i+1} - Loss: {running_loss/200:.4f}')
            running_loss = 0.0

# 6. Global Accuracy Assessment
print("\nTraining Complete. Assessing Global Accuracy...")
model.eval() # Switch to evaluation mode
correct = 0
total = 0

with torch.no_grad(): # Optimization: Memory management by disabling gradient tracking
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Final Accuracy: {100 * correct / total:.2f}%')

# 7. Serialization (Saving State Dict)
torch.save(model.state_dict(), 'cnn_model.pth')
print("State dictionary saved to 'cnn_model.pth'. Ready for deployment.")