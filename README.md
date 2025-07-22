# Yarn Color Detection

## Author: Federico Matthew Pratama - 233405001

---

## Yarn Color Detection Using Camera

To detect yarn color using your camera, run:

```bash
python camera_detect_yarn.py
```

Or, if you use Python 3:

```bash
python3 camera_detect_yarn.py
```

### How to Use

1. Point your camera at the yarn you want to detect.
2. Position the yarn in the center of the green box (the detection area).
3. The detected color information will appear at the bottom of the screen.
4. Press SPACE to save the detection result.
5. Press **q** to exit the program.

---

## Yarn Color Detection Using an Image File

To detect yarn color from an image file, run:

```bash
python input_image_detect_yarn.py
```

Or, if you use Python 3:

```bash
python3 input_image_detect_yarn.py
```

#### Example (using `orange_wol.png`):

```bash
python input_image_detect_yarn.py
```

Sample program output:

```
BASIC YARN COLOR DETECTOR
================================
Program ini akan menganalisis warna benang dari file gambar
Warna yang didukung: Brown, Green, Orange, Pink, Red, Cream, Yellow, Blue
================================

Masukkan path gambar (atau 'q' untuk keluar): ./orange_wol.png
Menganalisis gambar...

==================================================
HASIL ANALISIS WARNA BENANG
==================================================
Warna Terdeteksi: Orange
Kode: YC-007
Keyakinan: 99.0%

Nilai CIELAB: L=14.1, a=12.0, b=14.0
Nilai RGB: (57, 28, 16)
==================================================

Simpan hasil analisis? (y/n): n

Masukkan path gambar (atau 'q' untuk keluar):
```

### How to Use

1. Enter the path to your image file, for example `orange_wol.png`.
2. The program will display a summary of the detected yarn color.
3. To save the analysis, enter `y` for yes or `n` for no.
4. After saving (or skipping), you’ll be prompted for another image path; enter `q` to exit.

---

## Supported Yarn Colors

- Brown
- Green
- Orange
- Pink
- Red
- Cream
- Yellow
- Blue

---

Feel free to contribute or ask questions!
