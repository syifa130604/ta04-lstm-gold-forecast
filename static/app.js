const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) => n == null ? "-" :
  Number(n).toLocaleString("id-ID", { minimumFractionDigits: d, maximumFractionDigits: d });
const tgl = (s) => new Date(s + "T00:00:00").toLocaleDateString("id-ID", { day: "numeric", month: "long", year: "numeric" });

// Palet warna bertema emas
const WARNA = { emas: "#d4a017", emasTua: "#b8860b", coklat: "#3b2a0a", emasPudar: "rgba(212,160,23,0.16)" };

Chart.defaults.color = "#7a6a45";
Chart.defaults.font.family = '"Inter", system-ui, sans-serif';
Chart.defaults.borderColor = "rgba(184,134,11,0.15)";

async function ambil(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Terjadi kesalahan pada server.");
  return j;
}

const grafik = {};
function gambarGaris(id, labels, datasets) {
  if (grafik[id]) grafik[id].destroy();
  grafik[id] = new Chart($(id), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top", labels: { usePointStyle: true, boxWidth: 8 } },
        tooltip: {
          backgroundColor: "#2a1f08", titleColor: "#f6e27a", bodyColor: "#fff8e1", padding: 10,
          callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt(c.parsed.y)}` }
        }
      },
      scales: {
        x: { ticks: { maxTicksLimit: 8, maxRotation: 0 }, grid: { display: false } },
        y: { title: { display: true, text: "Harga Close" } }
      }
    }
  });
}
const garis = (label, data, warna, ekstra = {}) =>
  ({ label, data, borderColor: warna, backgroundColor: warna, borderWidth: 1.8, pointRadius: 0, tension: 0.1, ...ekstra });
const potong = (arr, n) => (n > 0 ? arr.slice(-n) : arr);

let dataHist, dataTest;

function tampilHist(n) {
  gambarGaris("chartHist", potong(dataHist.dates, n), [
    garis("Close", potong(dataHist.close, n), WARNA.emasTua, { fill: true, backgroundColor: WARNA.emasPudar }),
    garis("Rata-rata 30 hari", potong(dataHist.ma30, n), WARNA.coklat, { borderDash: [6, 4], borderWidth: 1.5 })
  ]);
}
function tampilTest(n) {
  gambarGaris("chartTest", potong(dataTest.dates, n), [
    garis("Harga aktual", potong(dataTest.aktual, n), WARNA.coklat),
    garis("Prediksi LSTM", potong(dataTest.prediksi, n), WARNA.emas, { borderWidth: 2.2 })
  ]);
}

function tampilMetrik(m) {
  const kartu = (judul, nilai, ket, kelas = "") =>
    `<div class="kartu ${kelas}"><small>${judul}</small><strong>${nilai}</strong><span>${ket}</span></div>`;
  $("metrik").innerHTML =
    kartu("RMSE LSTM", fmt(m.rmse), "pada data uji") +
    kartu("MAE LSTM", fmt(m.mae), "pada data uji") +
    kartu("MAPE LSTM", fmt(m.mape_persen) + "%", "kesalahan relatif") +
    kartu("Harga terakhir", fmt(m.harga_terakhir), tgl(m.tanggal_akhir), "sorot");
  $("subjudul").textContent =
    `Model LSTM dengan data ${tgl(m.tanggal_awal)} sampai ${tgl(m.tanggal_akhir)} (${m.jumlah_data.toLocaleString("id-ID")} hari perdagangan).`;

  const lebihBaik = m.rmse < m.rmse_baseline_naif;
  $("catatanModel").innerHTML =
    `RMSE LSTM <strong>${fmt(m.rmse)}</strong> dibanding tebakan sederhana "harga besok = harga hari ini" ` +
    `<strong>${fmt(m.rmse_baseline_naif)}</strong>. ` +
    (lebihBaik
      ? "Pada data uji ini LSTM lebih akurat daripada tebakan sederhana."
      : "Pada data uji ini LSTM belum mengalahkan tebakan sederhana. Garis prediksi cenderung lebih halus dan terlambat mengikuti perubahan harga.");
}

async function ramal(hari) {
  const pesan = $("pesanRamal");
  pesan.textContent = "Menghitung ramalan...";
  try {
    const r = await ambil(`/api/forecast?days=${hari}`);
    const labels = [...r.konteks_tanggal, ...r.ramalan_tanggal];
    const nKonteks = r.konteks_harga.length;
    const aktual = [...r.konteks_harga, ...Array(hari).fill(null)];
    const ramalan = Array(nKonteks - 1).fill(null).concat([r.konteks_harga[nKonteks - 1]], r.ramalan_harga);
    $("kotakRamal").hidden = false;
    gambarGaris("chartRamal", labels, [
      garis("Harga aktual (30 hari terakhir)", aktual, WARNA.coklat),
      garis("Ramalan LSTM", ramalan, WARNA.emas, {
        borderDash: [6, 4], borderWidth: 2.2, pointRadius: 4,
        pointBackgroundColor: WARNA.emas, pointBorderColor: WARNA.coklat
      })
    ]);
    const baris = r.ramalan_tanggal.map((t, i) => {
      const selisih = (r.ramalan_harga[i] - r.harga_terakhir) / r.harga_terakhir * 100;
      const kelas = selisih >= 0 ? "naik" : "turun";
      return `<tr><td>${tgl(t)}</td><td>${fmt(r.ramalan_harga[i])}</td>` +
             `<td class="${kelas}">${selisih >= 0 ? "+" : ""}${fmt(selisih)}%</td></tr>`;
    }).join("");
    $("tabelRamal").querySelector("tbody").innerHTML = baris;
    $("tabelRamal").hidden = false;
    pesan.textContent = `Ramalan ${hari} hari kerja setelah ${tgl(r.tanggal_terakhir)} (harga terakhir ${fmt(r.harga_terakhir)}).`;
  } catch (e) {
    pesan.textContent = "Gagal: " + e.message;
  }
}

document.addEventListener("click", (e) => {
  const tombol = e.target.closest(".rentang button");
  if (!tombol) return;
  const grup = tombol.parentElement;
  grup.querySelectorAll("button").forEach((b) => b.classList.remove("aktif"));
  tombol.classList.add("aktif");
  const n = Number(tombol.dataset.n);
  if (grup.dataset.target === "hist") tampilHist(n); else tampilTest(n);
});

$("formRamal").addEventListener("submit", (e) => {
  e.preventDefault();
  ramal(Number($("hari").value));
});

(async function mulai() {
  try {
    const [m, h, t] = await Promise.all([ambil("/api/metrics"), ambil("/api/history"), ambil("/api/test")]);
    dataHist = h; dataTest = t;
    tampilMetrik(m); tampilHist(0); tampilTest(0);
  } catch (e) {
    $("metrik").innerHTML = `<div class="kartu"><strong>Gagal memuat data</strong><span>${e.message}</span></div>`;
  }
})();