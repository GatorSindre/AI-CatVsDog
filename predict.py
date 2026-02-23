import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Define the same model architecture ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# --- 2. Set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Create model and load weights ---
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cat_dog_cnn.pth", map_location=device))
model.eval()  # important: set to evaluation mode

# --- 4. Load and transform image ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Load your test image
img = Image.open("test.png")
img = transform(img).unsqueeze(0)  # add batch dimension
img = img.to(device)

# --- 5. Predict ---
with torch.no_grad():  # disables gradient calculation, faster
    output = model(img)
    prediction = (output > 0.5).float()

# --- 6. Print result ---
if prediction.item() == 1.0:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")