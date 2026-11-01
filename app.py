"""Aplikasi web peramalan harga emas harian dengan LSTM (TA-04)."""
import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from predictor import LSTMPredictor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "gold_clean.csv"
MODEL_DIR = BASE_DIR / "model"
MAX_HARI = 14

app = Flask(__name__)
app.json.sort_keys = False

# ---- Muat semua aset satu kali saat aplikasi dijalankan ----
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
predictor = LSTMPredictor(MODEL_DIR / "lstm_weights.npz")
data = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
pred_test = pd.read_csv(MODEL_DIR / "pred_test.csv", parse_dates=["Date"])
with open(MODEL_DIR / "metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)
TIMESTEPS = int(metrics["timesteps"])
_kunci = threading.Lock()


def _daftar(seri, desimal=2):
    """Ubah deret angka ke list JSON yang aman (NaN menjadi None)."""
    return [None if pd.isna(v) else round(float(v), desimal) for v in seri]


def _tanggal(indeks):
    return [d.strftime("%Y-%m-%d") for d in indeks]


@app.route("/")
def beranda():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/api/metrics")
def api_metrics():
    info = dict(metrics)
    info["jumlah_data"] = int(len(data))
    info["tanggal_awal"] = data.index[0].strftime("%Y-%m-%d")
    info["tanggal_akhir"] = data.index[-1].strftime("%Y-%m-%d")
    info["harga_terakhir"] = round(float(data["Close"].iloc[-1]), 2)
    return jsonify(info)


@app.route("/api/history")
def api_history():
    ma30 = data["Close"].rolling(30).mean()
    return jsonify(dates=_tanggal(data.index), close=_daftar(data["Close"]), ma30=_daftar(ma30))


@app.route("/api/test")
def api_test():
    return jsonify(dates=_tanggal(pred_test["Date"]),
                   aktual=_daftar(pred_test["Aktual"]),
                   prediksi=_daftar(pred_test["Prediksi"]))


@app.route("/api/forecast")
def api_forecast():
    try:
        hari = int(request.args.get("days", 7))
    except ValueError:
        return jsonify(error="Parameter 'days' harus berupa bilangan bulat."), 400
    if not 1 <= hari <= MAX_HARI:
        return jsonify(error=f"Jumlah hari harus antara 1 dan {MAX_HARI}."), 400

    with _kunci:
        nilai = predictor.ramal(scaler, data["Close"].to_numpy(), hari, TIMESTEPS)
    terakhir = data.index[-1]
    tanggal = pd.bdate_range(terakhir + pd.Timedelta(days=1), periods=hari)  # hari kerja
    konteks = data["Close"].tail(30)
    return jsonify(
        harga_terakhir=round(float(data["Close"].iloc[-1]), 2),
        tanggal_terakhir=terakhir.strftime("%Y-%m-%d"),
        konteks_tanggal=_tanggal(konteks.index),
        konteks_harga=_daftar(konteks),
        ramalan_tanggal=_tanggal(tanggal),
        ramalan_harga=_daftar(nilai),
    )


if __name__ == "__main__":
    app.run(debug=True)