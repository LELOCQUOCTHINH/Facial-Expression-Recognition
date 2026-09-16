# -*- coding: utf-8 -*-
"""FER2013 Dataset (no h5py) — dùng .npz thay vì .h5

Giữ nguyên API:
- class FER2013(split='Training'|'PublicTest'|'PrivateTest', transform=None)
- __len__, __getitem__ trả về (PIL->transform(img), label)
"""

from __future__ import print_function
from PIL import Image
import numpy as np
import torch.utils.data as data
from pathlib import Path

class FER2013(data.Dataset):
    """`FER2013 Dataset` đọc từ data/data.npz (không dùng h5py).

    Args:
        split (str): 'Training', 'PublicTest', hoặc 'PrivateTest'
        transform (callable, optional): Hàm biến đổi PIL image (transforms.*)
        npz_path (str, optional): đường dẫn file npz (mặc định ./data/data.npz)
        memmap (bool, optional): dùng mmap_mode='r' để tiết kiệm RAM (mặc định True)
    """

    def __init__(self, split='Training', transform=None,
                 npz_path: str = './data/data.npz', memmap: bool = True):
        self.transform = transform
        self.split = split  # 'Training' | 'PublicTest' | 'PrivateTest'
        self.npz_path = Path(npz_path)

        if not self.npz_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {self.npz_path}. Hãy chạy script pre-process để tạo data.npz trước."
            )

        mmap_mode = 'r' if memmap else None
        npz = np.load(self.npz_path, mmap_mode=mmap_mode)

        if self.split == 'Training':
            X = npz['Training_pixel']    # shape: (N, 2304) uint8
            y = npz['Training_label']    # shape: (N,)
            N_expected = 28709  # theo FER2013 chuẩn; chỉ dùng để reshape, không bắt buộc khớp
        elif self.split == 'PublicTest':
            X = npz['PublicTest_pixel']
            y = npz['PublicTest_label']
            N_expected = 3589
        elif self.split == 'PrivateTest':
            X = npz['PrivateTest_pixel']
            y = npz['PrivateTest_label']
            N_expected = 3589
        else:
            raise ValueError("split phải là 'Training', 'PublicTest' hoặc 'PrivateTest'.")

        # X có shape (N, 2304). Reshape về (N, 48, 48)
        # Không bắt buộc dùng N_expected; ta reshape theo -1 cho an toàn:
        self.images = X.reshape((-1, 48, 48))
        self.labels = y.astype(np.int64, copy=False)
        self.N = self.images.shape[0]

    def __getitem__(self, index):
        """
        Returns:
            tuple: (image, target)
                - image: PIL Image (grayscale được nhân thành 3 kênh cho tương thích transform)
                - target: int (nhãn 0..6)
        """
        img = self.images[index]              # (48, 48), uint8
        target = int(self.labels[index])      # đảm bảo là int

        # chuyển (48,48) -> (48,48,1) -> (48,48,3) như file cũ
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)  # 3-channel grayscale
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.N
