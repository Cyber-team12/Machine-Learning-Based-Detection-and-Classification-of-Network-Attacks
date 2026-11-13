# Deteksi Serangan DDoS dengan LightGBM (CIC-DDoS2019)

Pipeline machine learning untuk mendeteksi trafik DDoS (reflection/amplification dan SYN flood) dari 44 fitur statistik per window 50 paket.

Kartika Cyber Hackathon 2026

## Hasil (test set temporal, 75.920 baris, threshold 0.2)

| Metrik | Nilai |
|---|---|
| Recall serangan | 99,9841% (12 lolos dari 75.265) |
| False positive benign | 0 dari 655 |
| ROC-AUC / PR-AUC | 1,0000000 / 1,0000000 |

Recall 100% di 13 dari 14 jenis serangan. Satu-satunya yang lolos: TFTP (12 dari 13.331 baris, recall 99,91%), semuanya di ujung akhir segmen serangan saat laju paket turun.

## Struktur

| File | Fungsi |
|---|---|
| `convert_csv.py` | PCAP -> 44 fitur per window 50 paket, label berdasarkan jam serangan |
| `clean_dedup.py` | Buang NaN/Inf, filter serangan ber-Rate < 1000, dedup hash, split waktu 70/15/15, cek leakage |
| `train.py` | LightGBM bertahap, bobot sampel adaptif, early stopping, sweep threshold |
| `eval.py` | Evaluasi test set, recall per jenis serangan, feature importance |
| `cek_lolos.py` | Analisis serangan yang lolos |
| `predict.py` | Demo inferensi: CSV fitur -> SERANGAN/BENIGN |
| `predict_pcap.py` | Inferensi langsung dari file PCAP/PCAPNG -> 44 fitur -> SERANGAN/BENIGN |
| `bedah_fp.py`, `bedah_fp2.py` | Analisis kontribusi fitur (TreeSHAP) dan uji *counterfactual* pada PCAP eksternal |
| `app.py` | Dashboard UI interaktif (Streamlit) untuk deteksi CSV/PCAP & visualisasi TreeSHAP |
| `dashboard_ddos.html` | Dashboard HTML mandiri untuk melihat hasil deteksi & evaluasi secara instan |
| `model_lgb_final_44fitur.txt` | Model final |
| `model_lgb_final_44fitur.txt.threshold.json` | Threshold operasional (0.2) |
| `fitur_terpilih_44.json` | Daftar 44 fitur |
| `sample_test.csv` | Sampel kecil data test untuk demo |
| `lotsofweb.pcapng`, `hasil_pcap.csv` | Sampel PCAP eksternal (trafik web normal) dan hasil ekstraksi prediksinya |

## Cara menjalankan

```bash
pip install -r requirements.txt
python predict.py sample_test.csv
python predict_pcap.py lotsofweb.pcapng
streamlit run app.py
python eval.py   # butuh csv_bersih/ (lihat bagian Data)
```

`predict.py` menghasilkan `hasil_prediksi.csv` dan tabel prediksi vs label asli.

## Data

Dataset: CIC-DDoS2019 (Canadian Institute for Cybersecurity), tidak disertakan karena ukuran PCAP besar.

Untuk mereproduksi dari PCAP:

1. Taruh PCAP di `data_mentah/` (nama diawali `SAT-`)
2. `python convert_csv.py`
3. `python clean_dedup.py`
4. `python train.py`
5. `python eval.py`

## Catatan desain

- **Split temporal per kelas**: data diurutkan waktu, 70% awal train, 15% val, 15% akhir test, sehingga test berisi segmen serangan paling akhir. Hash dicek tidak ada yang bocor antar split.
- **Label** ditentukan dari jendela jam serangan pasangan IP attacker-victim, bukan label per flow.
- **Filter Rate**: serangan ber-Rate < 1000 dibuang agar window yang tidak berperilaku seperti flood tidak diberi label serangan.
- **Nama fitur** mengikuti skema dataset: `Duration` sebenarnya rata-rata TTL, `Tot size` rata-rata ukuran paket.
- **Threshold 0.2** dipilih dari sweep di validation set (FNR+FPR minimum).

## Keterbatasan

- Benign di test hanya 655 baris (0,86%), jadi "FP 0" berbasis sampel kecil.
- `Srate` menyumbang ~86% gain, model sangat bergantung pada laju paket keluar.
- Hanya diuji pada satu dataset/lingkungan; generalisasi ke jaringan lain belum diuji.
- Window serangan yang laju paketnya turun (akhir TFTP) paling rawan lolos.