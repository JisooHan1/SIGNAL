# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os

# ===== Dataset Class =====
class GestureDataset(Dataset):
    def __init__(self, data_dir, label_path):
        self.X, self.y = [], []
        with open(label_path, 'r') as f:
            label_map = json.load(f)

        for label_name, label_idx in label_map.items():
            folder = os.path.join(data_dir, label_name)
            if not os.path.exists(folder):
                print(f"Warning: {folder} not found")
                continue

            for file in os.listdir(folder):
                if file.endswith(".npy"):
                    try:
                        arr = np.load(os.path.join(folder, file))
                        if arr.shape == (30, 63):
                            self.X.append(arr)
                            self.y.append(label_idx)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")

        if not self.X:
            raise ValueError("No valid data found!")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx])

# ===== Model Definition (BiLSTM) =====
class BiLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Using the output from the last time step

# ===== Training Function =====
def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# ===== Main Execution =====
if __name__ == "__main__":
    data_dir = "data/processed"
    label_path = "data/labels.json"
    save_path = "model/saved/gesture_model.pt"

    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset
    print("Loading processed data...")
    dataset = GestureDataset(data_dir, label_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create and train model
    with open(label_path, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    model = BiLSTM(input_size=63, hidden_size=128, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(num_epochs):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\n Model saved: {save_path}")
