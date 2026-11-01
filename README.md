# Peramalan Harga Emas Harian dengan LSTM (TA-04)

Proyek Tugas Akhir 04 Pembelajaran Mesin: peramalan harga penutupan emas harian
dengan LSTM dan aplikasi web Flask.

## Isi repositori

- `notebooks/` : notebook TP-04 dan TA-04 (EDA, preprocessing, model, evaluasi)
- `data/` : dataset asli dan data bersih
- `model/` : model Keras (.h5), scaler (.pkl), bobot untuk web (.npz), metrik
- `app.py`, `predictor.py` : aplikasi Flask dan inferensi LSTM berbasis NumPy
- `templates/`, `static/` : tampilan web

## Menjalankan lokal

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Buka http://127.0.0.1:5000

Catatan: aplikasi web memakai NumPy dengan bobot dari model Keras yang sudah
dilatih agar muat di server gratis 512 MB. Proyek ini untuk pembelajaran dan
bukan saran investasi.