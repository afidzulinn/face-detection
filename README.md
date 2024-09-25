# Deteksi dan Perbandingan Wajah dalam Rekaman DVR

Proyek ini mengimplementasikan sistem deteksi wajah dan masker dalam rekaman video DVR menggunakan model YOLOv8 yang telah dilatih sebelumnya dan library face_recognition.

## Fitur Utama

- Deteksi wajah dan masker dalam video DVR menggunakan model YOLOv8 custom (format ONNX).
- Perbandingan wajah yang terdeteksi dengan foto referensi menggunakan face_recognition.
- Mengembalikan waktu dalam format HH:MM:SS di mana wajah yang cocok atau masker ditemukan.
- Menghasilkan video output dengan penanda waktu dan bounding box untuk wajah dan masker yang terdeteksi.
- Menyimpan hasil deteksi dalam file teks untuk analisis lebih lanjut.


https://github.com/user-attachments/assets/be319933-d53a-46a6-81e0-03ed6a9449b2


## Instalasi

1. Clone repository ini:
   ```
   git clone https://github.com/afidzulinn/face-detection.git
   cd face-detection
   ```

2. Buat virtual environment (opsional tapi direkomendasikan):
   ```
   python -m venv venv
   source venv/bin/activate  # Untuk Unix
   venv\Scripts\activate  # Untuk Windows
   ```

3. Install dependensi:
   ```
   pip install -r requirements.txt
   ```

## Penggunaan

Jalankan program dengan command berikut:

```
python main.py
```

## Struktur Proyek

```
face-detection/
│
├── main.py      # Skrip utama
├── model/                      # Folder untuk model YOLOv8
│   └── best.onnx               # Model YOLOv8 dalam format ONNX
├── video/                      # Folder untuk video input
│   └── input_video.mp4         # Video DVR input
├── images/                     # Folder untuk gambar referensi
│   └── 1.jpg                   # Foto referensi wajah
├── result/                     # Folder untuk output
│   ├── video-result.mp4        # Video output hasil deteksi
│   └── timestamps.txt          # File teks hasil deteksi
├── requirements.txt            # Daftar dependensi
└── README.md                   # Dokumentasi proyek
```

## Cara Kerja

1. Model YOLOv8 (ONNX) digunakan untuk mendeteksi wajah dan masker dalam setiap frame video.
2. Wajah yang terdeteksi dibandingkan dengan foto referensi menggunakan face_recognition.
3. Waktu deteksi dan jenis deteksi (wajah atau masker) dicatat.
4. Video output dihasilkan dengan bounding box dan timestamp untuk deteksi.
5. Hasil deteksi disimpan dalam file teks dan ditampilkan di terminal.

## Hasil

Setelah menjalankan skrip, Anda akan mendapatkan:
1. Video output (`result/video-result.mp4`) dengan visualisasi deteksi.
2. File teks (`result/timestamps.txt`) berisi daftar waktu dan jenis deteksi.
3. Daftar deteksi yang ditampilkan di terminal.

## Screen Record Hasil Deteksi dan Result

Untuk mendapatkan screen record dari hasil deteksi:

1. Jalankan skrip seperti yang dijelaskan di atas.
2. Hasil video akan disimpan di (`result/video-result.mp4`).
3. Setelah selesai, putar video hasil (`result/video-result.mp4`).


```

Semua dependensi dapat diinstal menggunakan file `requirements.txt` yang disediakan.
