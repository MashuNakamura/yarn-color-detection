import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os

def load_color_db(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def rgb_to_lab(rgb):
    arr = np.uint8([[rgb]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0]
    L = lab[0] * 100.0 / 255.0
    a = float(lab[1]) - 128.0
    b = float(lab[2]) - 128.0
    return [L, a, b]

def find_nearest_color(lab, color_db):
    min_dist = float("inf")
    best_code = None
    best_name = None
    best_rgb = None
    for k, v in color_db.items():
        db_lab = v["lab"]
        dist = np.sqrt((lab[0]-db_lab[0])**2 + (lab[1]-db_lab[1])**2 + (lab[2]-db_lab[2])**2)
        if dist < min_dist:
            min_dist = dist
            best_code = v["code"]
            best_name = v["name"]
            best_rgb = v["rgb"]
    return best_code, best_name, best_rgb, min_dist

def get_roi_lab(img_pil):
    img = np.array(img_pil)
    h, w = img.shape[:2]
    x0, x1 = int(w*0.3), int(w*0.7)
    y0, y1 = int(h*0.3), int(h*0.7)
    roi = img[y0:y1, x0:x1]
    avg_rgb = np.mean(roi.reshape(-1, 3), axis=0)
    avg_rgb = [int(x) for x in avg_rgb]
    lab = rgb_to_lab(avg_rgb)
    return avg_rgb, lab

st.title("Yarn Color Identification (Benang) - Upload Gambar")
st.write("Upload gambar benang, aplikasi akan mendeteksi warna benang dan menampilkan hasil prediksi.")

color_json = "basic_colors_lab.json"
if not os.path.exists(color_json):
    st.error(f"File {color_json} tidak ditemukan! Upload dulu database warna atau gunakan contoh di bawah.")
else:
    color_db = load_color_db(color_json)

    uploaded_file = st.file_uploader("Upload gambar benang (JPG/PNG)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        # Slider untuk mengatur lebar tampilan gambar
        img_width = st.slider("Atur lebar gambar (px)", min_value=100, max_value=600, value=300)
        img_pil = Image.open(uploaded_file).convert("RGB")
        st.image(img_pil, caption="Gambar yang di-upload", width=img_width)
        avg_rgb, lab = get_roi_lab(img_pil)
        code, name, rgb, dist = find_nearest_color(lab, color_db)
        st.write(f"Warna rata-rata benang (ROI): RGB {avg_rgb}, LAB {lab}")
        st.write(f"Prediksi warna benang: **{name}**")
        st.write(f"Kode warna benang: `{code}`")
        st.write(f"RGB database: {rgb}, Jarak LAB: {dist:.2f}")
        st.markdown(
            f'<div style="width:60px;height:60px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});border:2px solid #333"></div>',
            unsafe_allow_html=True
        )