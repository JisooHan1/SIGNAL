# train.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from gesture_dataset.torch_dataset import GestureDataset, collate_fn
from gesture_dataset.model import GestureBiLSTM
from collections import Counter

# 📦 전처리된 데이터 불러오기
data = np.load("train_dataset.npz", allow_pickle=True)
sequences, labels = data['sequences'], data['labels']

# ✅ 클래스 분포 확인
print("Class distribution:", Counter(labels))

# PyTorch Dataset 구성
dataset = GestureDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureBiLSTM(input_dim=63, hidden_dim=128, output_dim=14).to(device)

# 손실 함수 & 옵티마이저
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct, total = 0, 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

    acc = correct / total * 100
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

# 모델 저장
torch.save(model.state_dict(), "gesture_bilstm.pt")
print("✅ 모델 저장 완료: gesture_bilstm.pt")
