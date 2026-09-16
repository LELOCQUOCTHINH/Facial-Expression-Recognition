# create data and label for FER2013 (no h5py)
# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
import csv, os
import numpy as np

CSV_PATH = 'data/fer2013.csv'
OUT_DIR  = 'data'
OUT_NPZ  = os.path.join(OUT_DIR, 'data.npz')

os.makedirs(OUT_DIR, exist_ok=True)

Training_x, Training_y = [], []
PublicTest_x, PublicTest_y = [], []
PrivateTest_x, PrivateTest_y = [], []

with open(CSV_PATH, 'r', newline='') as csvin:
    reader = csv.reader(csvin)
    header = next(reader, None)  # bỏ dòng header: [emotion,pixels,Usage]
    for row in reader:
        emotion = int(row[0])
        pixels  = np.fromstring(row[1], dtype=np.uint8, sep=' ')
        usage   = row[2]

        if usage == 'Training':
            Training_x.append(pixels)
            Training_y.append(emotion)
        elif usage == 'PublicTest':
            PublicTest_x.append(pixels)
            PublicTest_y.append(emotion)
        elif usage == 'PrivateTest':
            PrivateTest_x.append(pixels)
            PrivateTest_y.append(emotion)

# ép sang mảng numpy
Training_x = np.asarray(Training_x, dtype=np.uint8)      # shape: (N, 2304)
PublicTest_x = np.asarray(PublicTest_x, dtype=np.uint8)
PrivateTest_x = np.asarray(PrivateTest_x, dtype=np.uint8)

Training_y = np.asarray(Training_y, dtype=np.int64)      # shape: (N,)
PublicTest_y = np.asarray(PublicTest_y, dtype=np.int64)
PrivateTest_y = np.asarray(PrivateTest_y, dtype=np.int64)

print(Training_x.shape, PublicTest_x.shape, PrivateTest_x.shape)

# Lưu NPZ với CÙNG tên key như file .h5 để dễ thay thế trong code train
np.savez_compressed(
    OUT_NPZ,
    Training_pixel=Training_x,
    Training_label=Training_y,
    PublicTest_pixel=PublicTest_x,
    PublicTest_label=PublicTest_y,
    PrivateTest_pixel=PrivateTest_x,
    PrivateTest_label=PrivateTest_y
)

print("Saved:", OUT_NPZ)
