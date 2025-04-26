# gesture_dataset/torch_dataset.py

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class GestureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True)  # [B, T, 63]
    return padded_seqs, torch.tensor(labels, dtype=torch.long)
