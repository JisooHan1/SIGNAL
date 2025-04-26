# gesture_dataset/model.py

import torch.nn as nn

class GestureBiLSTM(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=256, output_dim=15):
        super(GestureBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3  # LSTM 내부 dropout
        )
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)                  # [B, T, 2H]
        out = out[:, -1, :]                    # 마지막 타임스텝만 사용
        out = self.dropout(out)                # dropout 추가
        out = self.relu(self.fc1(out))         # FC + ReLU
        return self.fc2(out)                   # 최종 출력 (softmax는 Loss 함수가 알아서 처리)





# # gesture_dataset/model.py

# import torch.nn as nn

# class GestureBiLSTM(nn.Module):
#     def __init__(self, input_dim=63, hidden_dim=128, output_dim=14):
#         super(GestureBiLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

#     def forward(self, x):
#         out, _ = self.lstm(x)              # [B, T, 2H]
#         out = self.dropout(out[:, -1, :])  # 마지막 타임스텝의 출력만 사용
#         return self.fc(out)
