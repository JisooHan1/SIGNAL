import torch
import torch.nn as nn

class GestureCNNBiLSTM(nn.Module):
    def __init__(self, input_dim=99, cnn_out=128, lstm_hidden=256, output_dim=15):
        super(GestureCNNBiLSTM, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_out,
            kernel_size=3,
            stride=1,
            padding=1
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
        # x shape: [batch_size, sequence_len, input_dim]
        x = x.permute(0, 2, 1)  # to [B, input_dim, T]
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # back to [B, T, cnn_out]
        x, _ = self.bilstm(x)
        x = x[:, -1, :]         # last timestep
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
