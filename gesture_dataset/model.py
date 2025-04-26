import torch
import torch.nn as nn

class GestureCNNBiLSTM(nn.Module):
    def __init__(self, input_dim=63, cnn_out=128, lstm_hidden=256, output_dim=15):
        super(GestureCNNBiLSTM, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_out,
            kernel_size=3,
            stride=1,
            padding=1  # 유지: 입력 길이 그대로 유지
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

        self.bilstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )

        self.fc1 = nn.Linear(lstm_hidden * 2, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: [B, T, 63] -> permute for Conv1d: [B, 63, T]
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)      # [B, cnn_out, T]
        x = self.relu(x)

        x = x.permute(0, 2, 1)  # [B, T, cnn_out] for LSTM
        out, _ = self.bilstm(x) # [B, T, 2H]
        out = out[:, -1, :]     # 마지막 타임스텝만
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)
