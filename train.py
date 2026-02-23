import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
])
print("Transformations defined.")

# Create datasets
training_set = torchvision.datasets.ImageFolder(
    root='./data/train',
    transform=transform
)
"""
testing_set = torchvision.datasets.ImageFolder(
    root='./data/test',
    transform=transform
)
"""
print("Datasets created.")

# Create data loaders; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)
# validation_loader = torch.utils.data.DataLoader(testing_set, batch_size=32, shuffle=False)
print("Data loaders created.")

print("Classes:", training_set.classes)

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
# print('Testing set has {} instances'.format(len(testing_set)))


import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolution: takes RGB image (3 channels) → 16 feature maps
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # Second convolution: 16 → 32 feature maps
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Max pooling layer: reduces image size by half
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layer: flatten 32x32x32 → 128 neurons
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        # Output layer: 1 neuron for binary classification
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Convolution → ReLU → Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 32 * 32 * 32)
        # Fully connected → ReLU
        x = F.relu(self.fc1(x))
        # Output → Sigmoid (gives probability 0–1)
        x = torch.sigmoid(self.fc2(x))
        return x
print("Model defined.")

model = SimpleCNN()               # our model
criterion = nn.BCELoss()          # binary cross-entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
print("Model, loss function, and optimizer created.")




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 15

try:
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in training_loader:
            # Move batch to GPU if available
            images, labels = images.to(device), labels.to(device)

            labels = labels.float().unsqueeze(1)  # make shape (batch_size,1) for BCE

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()  # convert bool → float 0.0 / 1.0
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(training_loader)
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        pass
    
except KeyboardInterrupt:
    print("\nKeyboard interrupt detected! Saving model...")
    torch.save(model.state_dict(), "cat_dog_cnn_backup.pth")
    print("Model saved! Exiting gracefully.")


torch.save(model.state_dict(), "cat_dog_cnn.pth")