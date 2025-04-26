# test.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from gesture_dataset.torch_dataset import GestureDataset, collate_fn
from gesture_dataset.model import GestureCNNBiLSTM
from collections import Counter

# 📦 전처리된 테스트 데이터 불러오기
data = np.load("test_dataset.npz", allow_pickle=True)
sequences, labels = data['sequences'], data['labels']

# ✅ 클래스 분포 확인 (optional)
print("Class distribution:", Counter(labels))

# PyTorch Dataset 구성
dataset = GestureDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureCNNBiLSTM(input_dim=63, cnn_out=128, lstm_hidden=128, output_dim=15).to(device)
model.load_state_dict(torch.load("gesture_cnn_bilstm.pt", map_location=device))
model.eval()

# 평가
correct, total = 0, 0
with torch.no_grad():
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        pred = outputs.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

acc = correct / total * 100
print(f"✅ Test Accuracy: {acc:.2f}%")
