from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os

# Define your model
class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--test_img', type=str, help='Path to test image')
args = parser.parse_args()

# ======== TRAINING ========
if args.train:
    data_path = "/Users/abassmac/Desktop/myAI/nums"
    if not os.path.isdir(data_path):
        raise FileNotFoundError("Training data folder not found.")

    dataset = ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DigitModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        running_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "custom_digit_model.pth")
    print("✅ Model saved as custom_digit_model.pth")

# ======== TESTING ========
elif args.test and args.test_img:
    if not os.path.isfile("custom_digit_model.pth"):
        raise FileNotFoundError("Trained model file not found.")

    if not os.path.isfile(args.test_img):
        raise FileNotFoundError("Test image not found.")

    model = DigitModel()
    model.load_state_dict(torch.load("custom_digit_model.pth"))
    model.eval()

    img = Image.open(args.test_img).convert("L")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        label = predicted.item()
        conf = confidence.item() * 100

    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {label} ({conf:.2f}% confidence)")
    plt.axis('off')
    plt.show()
else:
    print("Please specify --train or --test with --test_img.")
