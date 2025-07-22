import cv2
import numpy as np
import json

# Load database warna
def load_color_db(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

# Konversi RGB ke LAB
def rgb_to_lab(rgb):
    arr = np.uint8([[rgb]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0]
    L = lab[0] * 100.0 / 255.0
    a = float(lab[1]) - 128.0
    b = float(lab[2]) - 128.0
    return [L, a, b]

# Cari warna terdekat
def find_nearest_color(lab, color_db):
    min_dist = float("inf")
    best_code = None
    best_name = None
    best_rgb = None
    for v in color_db.values():
        db_lab = v["lab"]
        dist = np.sqrt((lab[0]-db_lab[0])**2 + (lab[1]-db_lab[1])**2 + (lab[2]-db_lab[2])**2)
        if dist < min_dist:
            min_dist = dist
            best_code = v["code"]
            best_name = v["name"]
            best_rgb = v["rgb"]
    return best_code, best_name, best_rgb, min_dist

# Ambil warna rata-rata dari ROI (tengah area gambar)
def get_roi_lab(img):
    h, w = img.shape[:2]
    x0, x1 = int(w*0.4), int(w*0.6)
    y0, y1 = int(h*0.4), int(h*0.6)
    roi = img[y0:y1, x0:x1]
    avg_bgr = np.mean(roi.reshape(-1, 3), axis=0)
    avg_bgr = [int(x) for x in avg_bgr]
    # Swap ke RGB karena OpenCV pakai BGR!
    avg_rgb = [avg_bgr[2], avg_bgr[1], avg_bgr[0]]
    lab = rgb_to_lab(avg_rgb)
    return avg_rgb, lab

def main():
    color_json = "camera_color.json"
    color_db = load_color_db(color_json)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Gagal membuka kamera. Pastikan webcam tersedia.")
        return

    print("Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ROI deteksi warna
        avg_rgb, lab = get_roi_lab(frame)
        code, name, rgb, dist = find_nearest_color(lab, color_db)

        # Gambar kotak ROI di frame
        h, w = frame.shape[:2]
        x0, x1 = int(w*0.4), int(w*0.6)
        y0, y1 = int(h*0.4), int(h*0.6)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,0), 2)

        # Gambar preview warna di frame
        preview_color = np.zeros((60,60,3), dtype=np.uint8)
        preview_color[:] = rgb[::-1]  # rgb ke BGR untuk OpenCV

        # Tulis hasil ke frame
        cv2.putText(frame, f"Prediksi: {name} ({code})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, f"RGB: {avg_rgb}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(frame, f"Database: {rgb}, Jarak: {dist:.2f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

        # Gabung preview warna ke frame (pojok kanan atas)
        frame[10:70, w-70:w-10] = preview_color

        cv2.imshow("Yarn Color Detection - Camera", frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()