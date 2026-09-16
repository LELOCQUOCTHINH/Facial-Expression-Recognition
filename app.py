# -*- coding: utf-8 -*-
"""
Streamlit App: Facial Expression Recognition (FER2013, VGG19)
Hiển thị kết quả theo hàng ngang (ảnh - biểu đồ - emoji)
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import streamlit as st

# import local modules
import transforms as transforms
from models import *

# ------------------------ DEVICE ------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cut_size = 44
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ------------------------ TRANSFORM ------------------------
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# ------------------------ MODEL LOADING ------------------------
@st.cache_resource
def load_model():
    model = VGG('VGG19')
    ckpt_path = os.path.join('FER2013_VGG19', 'PrivateTest_model.t7')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['net'], strict=True)
    model.to(device)
    model.eval()
    return model

net = load_model()

# ------------------------ UTILS ------------------------
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def preprocess_image(pil_img):
    """Chuyển ảnh sang grayscale (48x48x3) như FER2013"""
    gray = rgb2gray(np.array(pil_img))
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    return Image.fromarray(img)

def predict_expression(pil_img):
    inputs = transform_test(pil_img)
    ncrops, c, h, w = inputs.size()
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = net(inputs)
        outputs_avg = outputs.mean(0)
        score = F.softmax(outputs_avg, dim=0)
        _, predicted = torch.max(outputs_avg, dim=0)

    scores_np = score.detach().cpu().numpy()
    label = class_names[int(predicted.detach().cpu().item())]
    return label, scores_np

# ------------------------ STREAMLIT UI ------------------------
st.set_page_config(page_title="Facial Expression Recognition", layout="wide")
st.title("😃 Facial Expression Recognition (FER2013 - VGG19)")
st.markdown("Upload an image below to detect emotion:")

uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

# ------------------------ CUSTOM STYLE ------------------------
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    preprocessed_img = preprocess_image(raw_img)

    label, scores_np = predict_expression(preprocessed_img)

    st.markdown(f"## 🧠 Predicted Emotion: **{label}**")

    # 3 cột hiển thị ngang
    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col1:
        st.image(raw_img, caption="Input Image", use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']
        ind = np.arange(len(class_names))
        ax.bar(ind, scores_np, color=colors)
        ax.set_xticks(ind)
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_ylabel("Classification Score")
        ax.set_title("Classification results")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col3:
        emoji_path = f'images/emojis/{label}.png'
        if os.path.exists(emoji_path):
            st.image(emoji_path, caption="Emoji Expression", use_container_width=True)
        else:
            st.warning("Emoji image not found for this emotion.")

else:
    st.info("Please upload a face image to start prediction 😊")
