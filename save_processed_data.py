import numpy as np
from gesture_dataset.dataset_loader import create_dataset

ANNOT_PATH = "ipn_hand/annotation/Annot_TrainList.txt"
FRAMES_DIR = "ipn_hand/frames"

# 이미 로딩된 데이터가 있으면 여기 생략 가능
train_sequences, train_labels = create_dataset(ANNOT_PATH, FRAMES_DIR, max_sequences=None)

# ✅ 수정된 저장 부분
np.savez_compressed(
    "train_dataset.npz",
    sequences=np.array(train_sequences, dtype=object),
    labels=np.array(train_labels)
)

print("✅ 전처리 데이터 저장 완료: train_dataset.npz")
