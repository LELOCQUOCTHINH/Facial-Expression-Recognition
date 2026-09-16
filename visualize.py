"""
Visualize result for a single test image using CK+ checkpoint
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F

import transforms as transforms   # transforms của repo
from skimage import io
from skimage.transform import resize
from models import *              # VGG, ResNet18, ...

# ==== cấu hình cần chỉnh theo run của bạn ====
MODEL = 'VGG19'               # 'VGG19' hoặc 'Resnet18' (phải khớp opt.model lúc train)
FOLD  = 2                     # fold đã dùng khi train (1-based)
IMAGE_PATH = 'images/5.png'   # ảnh cần inference
# =============================================

# device (tự động CPU/GPU)
device = torch.device('cpu')  # ép CPU


cut_size = 44

# TenCrop test-time augmentation y như lúc evaluate
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# CK+ 7 lớp — khớp thứ tự index trong code train của bạn
class_names = ['anger','disgust','fear','happy','sadness','surprise','contempt']


def rgb2gray(rgb):
    # NTSC coefficients
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# 0) Kiểm tra đầu vào
assert os.path.isfile(IMAGE_PATH), f"Không tìm thấy ảnh: {IMAGE_PATH}"

# 1) Đọc & chuẩn hoá ảnh giống pipeline train/eval
raw_img = io.imread(IMAGE_PATH)                         # RGB/BGR -> coi như RGB
if raw_img.ndim == 2:                                   # nếu ảnh đã grayscale
    gray = raw_img.astype(np.float64) / 255.0
else:
    gray = rgb2gray(raw_img)                            # (H, W)
gray = resize(gray, (48, 48), mode='symmetric', anti_aliasing=True)  # (48,48), float64 [0..1]
gray = (gray * 255.0).astype(np.uint8)                  # về uint8 như trong repo

# Lặp kênh: nhiều model trong repo kỳ vọng 3 kênh
img = gray[:, :, np.newaxis]                            # (48,48,1)
img = np.concatenate((img, img, img), axis=2)           # (48,48,3)
img = Image.fromarray(img)
inputs = transform_test(img)                            # (10, C, 44, 44)

# 2) Build model & load checkpoint CK+
if MODEL == 'VGG19':
    net = VGG('VGG19')
elif MODEL.lower() in ['resnet18', 'resnet-18', 'resnet_18', 'Resnet18']:
    net = ResNet18()
else:
    raise ValueError(f'Unknown MODEL: {MODEL}')

# Thư mục checkpoint đúng với logic train: "{dataset}_{model}/{fold}/Test_model.t7"
ckpt_dir = os.path.join(f'CK+_{MODEL}', str(FOLD))
print(ckpt_dir)
ckpt_path = os.path.join(ckpt_dir, 'Test_model.t7')
print(ckpt_path)

assert os.path.isfile(ckpt_path), f'Không tìm thấy checkpoint: {ckpt_path}.\n' \
                                  f'Vui lòng kiểm tra cấu trúc: CK+_{MODEL}/{FOLD}/Test_model.t7'

# Nạp checkpoint an toàn (kể cả khác key hoặc có 'module.' prefix)
ckpt = torch.load(ckpt_path, map_location=device)
state = None
for k in ['net', 'state_dict', 'model', 'model_state', 'model_state_dict']:
    if isinstance(ckpt, dict) and k in ckpt:
        state = ckpt[k]
        break
if state is None:
    # một số repo lưu trực tiếp state_dict
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        raise RuntimeError(f"Không tìm thấy state_dict hợp lệ trong checkpoint: keys={list(ckpt.keys())}")

# Strip 'module.' nếu có (khi train DataParallel)
from collections import OrderedDict
new_state = OrderedDict()
for k, v in state.items():
    new_k = k.replace('module.', '') if k.startswith('module.') else k
    new_state[new_k] = v

missing, unexpected = net.load_state_dict(new_state, strict=False)
if missing:
    print("[Cảnh báo] Thiếu tham số trong checkpoint:", missing)
if unexpected:
    print("[Cảnh báo] Tham số thừa trong checkpoint:", unexpected)

net.to(device)
net.eval()

# 3) Inference với TenCrop: gộp 10 crop -> model -> average logits
with torch.no_grad():    
    ncrops, c, h, w = inputs.size()                     # (10, C, 44, 44)
    inputs = inputs.view(-1, c, h, w).to(device)        # (10, C, 44, 44)
    outputs = net(inputs)                               # (10, num_classes)
    print(outputs)
    # Average logits theo 10 crops để ổn định dự đoán
    outputs_avg = outputs.view(ncrops, -1).mean(0)      # (num_classes,)
    score = F.softmax(outputs_avg, dim=0).detach().cpu().numpy()
    predicted = int(outputs_avg.argmax().detach().cpu().numpy())

# 4) Vẽ kết quả
plt.rcParams['figure.figsize'] = (13.5, 5.5)

# Ảnh gốc
axes = plt.subplot(1, 3, 1)
plt.imshow(raw_img)
plt.xlabel('Input Image', fontsize=16)
axes.set_xticks([]); axes.set_yticks([])
plt.tight_layout()

# Biểu đồ xác suất
plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)
plt.subplot(1, 3, 2)
ind = 0.1 + 0.6 * np.arange(len(class_names))
width = 0.4
for i in range(len(class_names)):
    plt.bar(ind[i], float(score[i]), width)
plt.title("Classification results", fontsize=20)
plt.xlabel("Expression Category", fontsize=16)
plt.ylabel("Classification Score", fontsize=16)
plt.xticks(ind, class_names, rotation=45, fontsize=14)

# Emoji lớp dự đoán (nếu bạn có bộ emoji CK+)
axes = plt.subplot(1, 3, 3)
emoji_path = os.path.join('images', 'emojis', f'{class_names[predicted]}.png')
if os.path.isfile(emoji_path):
    emojis_img = io.imread(emoji_path)
    plt.imshow(emojis_img)
    plt.xlabel('Emoji Expression', fontsize=16)
else:
    plt.text(0.5, 0.5, class_names[predicted], ha='center', va='center', fontsize=24)
    plt.xlabel('Predicted', fontsize=16)
axes.set_xticks([]); axes.set_yticks([])
plt.tight_layout()

# Lưu ảnh kết quả
os.makedirs(os.path.join('images', 'results'), exist_ok=True)
out_path = os.path.join('images', 'results', 'ck_infer.png')
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Predicted: {class_names[predicted]}")
print(f"Saved visualization to: {out_path}")
