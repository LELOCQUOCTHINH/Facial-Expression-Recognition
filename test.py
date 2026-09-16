# -*- coding: utf-8 -*-
"""Đọc thử file data.npz (FER2013 đã pre-process)."""

import numpy as np
from pathlib import Path

NPZ_PATH = Path("data/data.npz")

def main():
    if not NPZ_PATH.exists():
        print(f"Không tìm thấy file: {NPZ_PATH.resolve()}")
        return

    npz = np.load(NPZ_PATH, mmap_mode='r')

    print("Các keys trong file:", list(npz.keys()))

    for key in npz.keys():
        arr = npz[key]
        print(f"{key:20s} shape={arr.shape}, dtype={arr.dtype}")

    # thử in 1 phần tử Training
    train_x = npz["Training_pixel"]
    train_y = npz["Training_label"]
    print("\nVí dụ sample 0:")
    print("pixels (length):", len(train_x[0]))
    print("label:", train_y[0])
    print("reshape thành (48,48):", train_x[0].reshape(48,48).shape)

if __name__ == "__main__":
    main()
