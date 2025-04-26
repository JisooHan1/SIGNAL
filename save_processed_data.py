import numpy as np
import os
from gesture_dataset.dataset_loader import create_dataset

FRAMES_DIR = "ipn_hand/frames"

# ✅ 학습 데이터
if os.path.exists("train_dataset.npz"):
    print("✅ train_dataset.npz 이미 존재 — 건너뜀")
else:
    train_sequences, train_labels = create_dataset(
        "ipn_hand/annotation/Annot_TrainList.txt",
        FRAMES_DIR,
        max_sequences=None
    )
    np.savez_compressed(
        "train_dataset.npz",
        sequences=np.array(train_sequences, dtype=object),
        labels=np.array(train_labels)
    )
    print(f"✅ 학습 데이터 저장 완료: train_dataset.npz ({len(train_sequences)} 시퀀스)")

# ✅ 테스트 데이터
if os.path.exists("test_dataset.npz"):
    print("✅ test_dataset.npz 이미 존재 — 건너뜀")
else:
    test_sequences, test_labels = create_dataset(
        "ipn_hand/annotation/Annot_TestList.txt",
        FRAMES_DIR,
        max_sequences=None
    )
    np.savez_compressed(
        "test_dataset.npz",
        sequences=np.array(test_sequences, dtype=object),
        labels=np.array(test_labels)
    )
    print(f"✅ 테스트 데이터 저장 완료: test_dataset.npz ({len(test_sequences)} 시퀀스)")
