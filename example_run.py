# example_run.py

from gesture_dataset.dataset_loader import create_dataset
from torch.utils.data import DataLoader
from gesture_dataset.torch_dataset import GestureDataset, collate_fn

ANNOT_PATH = "ipn_hand/annotation/Annot_TrainList.txt"
FRAMES_DIR = "ipn_hand/frames"

sequences, labels = create_dataset(ANNOT_PATH, FRAMES_DIR, max_sequences=None)

print(f"✅ Loaded {len(sequences)} sequences")
print(f"➡️ First sequence shape: {sequences[0].shape}")
print(f"➡️ First label: {labels[0]}")

# 이전까지 불러온 sequences, labels 사용
dataset = GestureDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

for batch_x, batch_y in dataloader:
    print("X shape:", batch_x.shape)  # [B, T, 63]
    print("Y shape:", batch_y.shape)  # [B]
    break
