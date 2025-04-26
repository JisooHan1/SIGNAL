# gesture_dataset/model.py

import torch.nn as nn

class GestureBiLSTM(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, output_dim=14):
        super(GestureBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)              # [B, T, 2H]
        out = self.dropout(out[:, -1, :])  # 마지막 타임스텝의 출력만 사용
        return self.fc(out)
