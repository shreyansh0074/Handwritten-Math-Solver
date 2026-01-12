import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# =========================
# CONFIG
# =========================
DATASET_PATH = r"D:\Codes\Python\projects\dataset"
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
MODEL_SAVE_PATH = "math_cnn.pth"

# =========================
# CLASS MAP
# =========================
class_map = {
    '0':0, '1':1, '2':2, '3':3, '4':4,
    '5':5, '6':6, '7':7, '8':8, '9':9,
    'add':10, 'sub':11, 'mul':12, 'div':13
}

idx_to_class = {v:k for k,v in class_map.items()}

NUM_CLASSES = 14

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# DATASET
# =========================
class MathDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        valid_ext = (".png", ".jpg", ".jpeg", ".bmp")

        for folder in class_map:
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue

            for img_name in os.listdir(folder_path):

                # 🔥 HARD SKIP ANY HIDDEN OR NON-IMAGE FILE
                if img_name.startswith("."):
                    print("Skipping hidden:", img_name)
                    continue

                if not img_name.lower().endswith(valid_ext):
                    print("Skipping non-image:", img_name)
                    continue

                img_path = os.path.join(folder_path, img_name)
                self.samples.append((img_path, class_map[folder]))

        print("Total samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            img = Image.open(img_path).convert("L")
        except:
            # 🔥 If any corrupt file still exists, skip it safely
            new_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(new_idx)

        if self.transform:
            img = self.transform(img)

        return img, label

# =========================
# MODEL
# =========================
class MathCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # 28x28 -> 26x26
        x = self.pool(x)              # 26x26 -> 13x13
        x = F.relu(self.conv2(x))     # 13x13 -> 11x11
        x = self.pool(x)              # 11x11 -> 5x5
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =========================
# TRAINING
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = MathDataset(DATASET_PATH, transform)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MathCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f} Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved as", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
