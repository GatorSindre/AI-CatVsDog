from PIL import Image
import os
import glob
import pyperclip
import keyboard
from notifypy import Notify
from pathlib import Path
import sys
import pyautogui
import time
from gtts import gTTS

import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cat_dog_cnn.pth", map_location=device))
model.eval()  # important: set to evaluation mode

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


## Used to access temporary files from .exe cache
base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
## Get direct file path's
icon_path = os.path.join(base_path, "notification_media", "checkmark.png")
audio_path = os.path.join(base_path, "notification_media", "sound_cue.wav")

## Get folder for screenshots
screenshots_folder = str(Path.home() / "Pictures" / "Screenshots")

## Notifications Setup    :TD: Make audio cue for it
notification = Notify()
notification.title = "DogOrCat"
notification.message = "AI finished"
notification.icon = icon_path
notification.audio = audio_path

def Predict_Screenshot():

    ## Gets folder of screenshots
    screenshots = glob.glob(os.path.join(screenshots_folder, "*.png"))

    if not screenshots:
        print("no screenshots found")
        return

    ## Returns the newest screenshot
    newest_screenshot = max(screenshots, key=os.path.getctime)

    img = Image.open(newest_screenshot)  # open the image file
    img = transform(img).unsqueeze(0)  # add batch dimension
    img = img.to(device)

    # --- 5. Predict ---
    with torch.no_grad():  # disables gradient calculation, faster
        output = model(img)
        prediction = (output > 0.5).float()

    prediction_final = ""
    if prediction.item() == 1.0:
        prediction_final = "Dog"

    else:
        prediction_final = "Cat"
    
    print(f"Prediction: {prediction_final}")

    pyperclip.copy(prediction_final) ## Copy result to clipboard

## Run in background with low cpu using an event

keyboard.add_hotkey("ctrl+shift+alt+i", Predict_Screenshot)

print("CatOrDog Initialized")

keyboard.wait()