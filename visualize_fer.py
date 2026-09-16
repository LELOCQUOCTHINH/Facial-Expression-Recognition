# -*- coding: utf-8 -*-
"""
visualize results for test image (CPU/GPU-safe, modern PyTorch)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from skimage import io
from skimage.transform import resize

import transforms as transforms   # transforms trong repo
from models import *              # VGG, ResNet18, ...

# ---- device: tự chọn CUDA nếu có, nếu không thì CPU ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# ---- đọc ảnh & tiền xử lý ----
raw_img = io.imread('images/1.jpg')  # đổi path nếu cần
gray = rgb2gray(raw_img)
gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
img = gray[:, :, np.newaxis]
img = np.concatenate((img, img, img), axis=2)
img = Image.fromarray(img)
inputs = transform_test(img)  # Tensor shape: (10, C, H, W)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---- build & load model ----
net = VGG('VGG19')  # hoặc ResNet18 nếu bạn train bằng ResNet18
ckpt_path = os.path.join('FER2013_VGG19', 'PrivateTest_model.t7')

# quan trọng: map_location=device để load checkpoint dù được save trên GPU
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['net']
net.load_state_dict(state_dict, strict=True)
net.to(device)
net.eval()

# ---- inference ----
# inputs: (ncrops, C, H, W) -> (ncrops, C, H, W) (đã đúng)
ncrops, c, h, w = inputs.size()
inputs = inputs.to(device)

with torch.no_grad():
    outputs = net(inputs)                       # (ncrops, num_classes)
    outputs_avg = outputs.mean(0)               # avg over crops -> (num_classes,)
    score = F.softmax(outputs_avg, dim=0)       # dim=0 vì 1D tensor
    _, predicted = torch.max(outputs_avg, dim=0)

# ---- plot ----
plt.rcParams['figure.figsize'] = (13.5, 5.5)

axes = plt.subplot(1, 3, 1)
plt.imshow(raw_img)
plt.xlabel('Input Image', fontsize=16)
axes.set_xticks([]); axes.set_yticks([])
plt.tight_layout()

plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

plt.subplot(1, 3, 2)
ind = 0.1 + 0.6 * np.arange(len(class_names))
width = 0.4
# giữ nguyên color_list nếu bạn muốn, hoặc để mặc định matplotlib
color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
scores_np = score.detach().cpu().numpy()
for i in range(len(class_names)):
    plt.bar(ind[i], float(scores_np[i]), width, color=color_list[i])
plt.title("Classification results", fontsize=20)
plt.xlabel("Expression Category", fontsize=16)
plt.ylabel("Classification Score", fontsize=16)
plt.xticks(ind, class_names, rotation=45, fontsize=14)

axes = plt.subplot(1, 3, 3)
emoji_idx = int(predicted.detach().cpu().item())
emojis_img = io.imread(f'images/emojis/{class_names[emoji_idx]}.png')
plt.imshow(emojis_img)
plt.xlabel('Emoji Expression', fontsize=16)
axes.set_xticks([]); axes.set_yticks([])
plt.tight_layout()

os.makedirs('images/results', exist_ok=True)
plt.savefig(os.path.join('images/results', 'l.png'))
plt.close()

print("The Expression is %s" % class_names[emoji_idx])
