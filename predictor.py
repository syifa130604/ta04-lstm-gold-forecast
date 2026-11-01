"""Inferensi LSTM murni NumPy.

Bobot diambil dari model Keras yang sudah dilatih (model.get_weights() disimpan
ke lstm_weights.npz). Dengan cara ini aplikasi web tidak membutuhkan TensorFlow
saat berjalan, sehingga ringan dan muat di server gratis dengan RAM 512 MB.
Arsitektur: LSTM(50, return_sequences) -> Dropout -> LSTM(50) -> Dropout
-> Dense(25) -> Dense(1). Dropout tidak aktif saat inferensi.
"""
import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm(x, kernel, recurrent, bias, return_sequences):
    """Satu layer LSTM Keras. x berbentuk (batch, timesteps, fitur)."""
    batch, steps, _ = x.shape
    units = recurrent.shape[0]
    h = np.zeros((batch, units), dtype=np.float32)
    c = np.zeros((batch, units), dtype=np.float32)
    keluaran = []
    for t in range(steps):
        z = x[:, t, :] @ kernel + h @ recurrent + bias
        i = _sigmoid(z[:, :units])               # input gate
        f = _sigmoid(z[:, units:2 * units])      # forget gate
        g = np.tanh(z[:, 2 * units:3 * units])   # kandidat cell state
        o = _sigmoid(z[:, 3 * units:])           # output gate
        c = f * c + i * g
        h = o * np.tanh(c)
        keluaran.append(h)
    return np.stack(keluaran, axis=1) if return_sequences else h


class LSTMPredictor:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        w = [data[f"arr_{i}"].astype(np.float32) for i in range(len(data.files))]
        if len(w) != 10:
            raise ValueError(f"Jumlah array bobot {len(w)}, seharusnya 10.")
        (self.k1, self.r1, self.b1,
         self.k2, self.r2, self.b2,
         self.wd1, self.bd1, self.wd2, self.bd2) = w

    def predict(self, x):
        """x: array (batch, timesteps, 1) data ter-scaling. Mengembalikan (batch, 1)."""
        x = np.asarray(x, dtype=np.float32)
        h1 = _lstm(x, self.k1, self.r1, self.b1, return_sequences=True)
        h2 = _lstm(h1, self.k2, self.r2, self.b2, return_sequences=False)
        d1 = h2 @ self.wd1 + self.bd1
        return d1 @ self.wd2 + self.bd2

    def ramal(self, scaler, harga_close, langkah=7, timesteps=60):
        """Ramal 'langkah' hari ke depan secara berantai: hasil prediksi
        dipakai sebagai input hari berikutnya. Mengembalikan harga asli."""
        ekor = pd.DataFrame({"Close": np.asarray(harga_close, dtype=float)[-timesteps:]})
        window = scaler.transform(ekor).astype(np.float32).reshape(1, timesteps, 1)
        hasil = []
        for _ in range(langkah):
            p = float(self.predict(window)[0, 0])
            hasil.append(p)
            window = np.concatenate(
                [window[:, 1:, :], np.array(p, dtype=np.float32).reshape(1, 1, 1)], axis=1)
        return scaler.inverse_transform(np.array(hasil).reshape(-1, 1)).ravel()