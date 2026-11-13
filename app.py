import json
import os
import tempfile
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st

st.set_page_config(
    page_title="Sistem Deteksi DDoS LightGBM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: Warna Putih Cream & Cokelat Susu Kalem
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FAF7F2;
        color: #2C221E;
    }
    section[data-testid="stSidebar"] {
        background-color: #EFE9E0;
        border-right: 1px solid #DCD2C4;
    }
    div[data-testid="stMetric"] {
        background-color: #F4EFE6;
        border: 1px solid #DFD5C7;
        padding: 14px 16px;
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] {
        color: #6E5D4F !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #2C221E !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #EFE9E0;
        border: 1px solid #DCD2C4;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        color: #5C4B3C;
    }
    .stTabs [aria-selected="true"] {
        background-color: #8C6239 !important;
        color: #FFFFFF !important;
        border-color: #8C6239 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DEFAULT = "model_lgb_final_44fitur.txt"
FITUR_JSON = "fitur_terpilih_44.json"
AUDIT_JSON = "csv_bersih/audit_clean_summary.json"


@st.cache_resource
def load_model_and_features(model_path: str):
    booster = lgb.Booster(model_file=model_path)
    with open(FITUR_JSON, "r", encoding="utf-8") as f:
        fitur = json.load(f)
    tpath = model_path + ".threshold.json"
    default_th = 0.2
    if os.path.exists(tpath):
        with open(tpath, "r", encoding="utf-8") as tf:
            default_th = float(json.load(tf).get("threshold", 0.2))
    return booster, fitur, default_th


def load_audit_summary():
    if os.path.exists(AUDIT_JSON):
        with open(AUDIT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


BENCHMARK_PER_KELAS = [
    {"Jenis Serangan": "DrDoS_DNS", "Total Test": 16760, "Terdeteksi": 16760, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "TFTP", "Total Test": 13331, "Terdeteksi": 13319, "Lolos (FN)": 12, "Recall (%)": 99.910},
    {"Jenis Serangan": "DrDoS_NTP", "Total Test": 12082, "Terdeteksi": 12082, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "DrDoS_SNMP", "Total Test": 7648, "Terdeteksi": 7648, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "DrDoS_LDAP", "Total Test": 5845, "Terdeteksi": 5845, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "NetBIOS", "Total Test": 5570, "Terdeteksi": 5570, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "DrDoS_MSSQL", "Total Test": 2788, "Terdeteksi": 2788, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "MSSQL", "Total Test": 2701, "Terdeteksi": 2701, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "LDAP", "Total Test": 2610, "Terdeteksi": 2610, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "DrDoS_UDP", "Total Test": 2251, "Terdeteksi": 2251, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "DrDoS_SSDP", "Total Test": 1890, "Terdeteksi": 1890, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "DrDoS_NetBIOS", "Total Test": 1081, "Terdeteksi": 1081, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "UDP", "Total Test": 448, "Terdeteksi": 448, "Lolos (FN)": 0, "Recall (%)": 100.000},
    {"Jenis Serangan": "Syn", "Total Test": 260, "Terdeteksi": 260, "Lolos (FN)": 0, "Recall (%)": 100.000},
]


def main():
    booster, fitur_44, default_th = load_model_and_features(MODEL_DEFAULT)

    with st.sidebar:
        st.title("Deteksi DDoS LightGBM")
        st.caption("Kartika Cyber Hackathon 2026")
        st.markdown("---")
        st.subheader("Parameter Inferensi")
        threshold = st.slider(
            "Threshold Skor Serangan",
            min_value=0.05,
            max_value=0.95,
            value=float(default_th),
            step=0.05,
            help="Window dengan skor probabilitas >= threshold diklasifikasikan sebagai SERANGAN (Default: 0.20).",
        )
        label_asli_pcap = st.selectbox(
            "Ground Truth (Label Asli) untuk PCAP Tanpa Label:",
            [
                "Semua BENIGN (Contoh: lotsofweb.pcapng)",
                "Semua SERANGAN (Jika menguji PCAP serangan murni)",
                "Tidak Diketahui (Tanpa evaluasi akurasi)",
            ],
            help="File PCAP mentah tidak memiliki kolom Label. Pilih asumsi label asli untuk menghitung Akurasi, FPR, atau Recall.",
        )
        max_pkts_pcap = st.number_input(
            "Batas Paket PCAP",
            min_value=1000,
            max_value=500000,
            value=50000,
            step=5000,
        )
        st.markdown("---")
        st.markdown(
            """
            **Spesifikasi Model**
            - Algoritma: LightGBM (GBDT)
            - Input: 44 Fitur / Window 50 Paket
            - Dataset: CIC-DDoS2019
            - Pembagian Data: Temporal 70/15/15
            - Threshold Optimal: 0.20
            """
        )

    st.title("Sistem Deteksi Serangan DDoS Berbasis Window 50 Paket")
    st.markdown(
        "Pendeteksian trafik DDoS (Reflection/Amplification dan SYN Flood) menggunakan ekstraksi 44 fitur statistik per window 50 paket IP dan model LightGBM."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall Serangan (Test Set)", "99.9841%", "75.253 / 75.265 terdeteksi")
    c2.metric("False Positive Benign", "0.0000%", "0 / 655 salah deteksi")
    c3.metric("ROC-AUC / PR-AUC", "1.0000000", "14 Jenis Serangan DDoS")
    c4.metric("Skema Fitur", "44 Fitur", "Per Window 50 Paket")

    st.markdown("---")

    tab_deteksi, tab_xai, tab_evaluasi = st.tabs([
        "1. Deteksi & Evaluasi Data (PCAP / CSV)",
        "2. Analisis Kontribusi Fitur (SHAP & Simulasi)",
        "3. Hasil Benchmark Test Set & Audit Data",
    ])

    # =========================================================================
    # TAB 1: DETEKSI INTERAKTIF
    # =========================================================================
    with tab_deteksi:
        st.subheader("Pengujian Model pada File PCAP atau CSV")
        sumber = st.radio(
            "Pilih Sumber Data:",
            [
                "Uji File PCAP Lokal (lotsofweb.pcapng)",
                "Uji Hasil Test Set Berlabel (hasil_prediksi.csv — 75.920 baris)",
                "Uji Sampel Test Set (sample_test.csv)",
                "Upload File PCAP / PCAPNG Baru",
                "Upload File CSV (44 Fitur)",
            ],
            horizontal=True,
        )

        df_input = None
        nama_sumber = ""

        if sumber.startswith("Uji File PCAP Lokal"):
            mode_lots = st.radio(
                "Metode pembacaan lotsofweb.pcapng:",
                [
                    "Muat cepat dari hasil ekstraksi (hasil_pcap.csv — 309 window)",
                    "Ekstrak ulang langsung dari file biner lotsofweb.pcapng (Scapy)",
                ],
                horizontal=True,
            )
            if mode_lots.startswith("Muat cepat") and os.path.exists("hasil_pcap.csv"):
                df_input = pd.read_csv("hasil_pcap.csv")
                nama_sumber = "lotsofweb.pcapng (309 window)"
            elif os.path.exists("lotsofweb.pcapng"):
                from predict_pcap import ekstrak
                with st.spinner("Mengekstrak 44 fitur per window 50 paket dari lotsofweb.pcapng..."):
                    rows, n_pkts = ekstrak("lotsofweb.pcapng", window_size=50, max_pkts=int(max_pkts_pcap))
                    if rows:
                        df_input = pd.DataFrame(rows)
                        if "Label" in df_input.columns:
                            df_input = df_input.drop(columns=["Label"])
                        nama_sumber = f"lotsofweb.pcapng ({n_pkts:,} paket IP, {len(df_input):,} window)"
            else:
                st.error("File lotsofweb.pcapng tidak ditemukan.")

        elif sumber.startswith("Uji Hasil Test Set Berlabel"):
            if os.path.exists("hasil_prediksi.csv"):
                df_input = pd.read_csv("hasil_prediksi.csv")
                nama_sumber = "hasil_prediksi.csv (Test Set Lengkap 75.920 baris)"
            else:
                st.error("File hasil_prediksi.csv tidak ditemukan.")

        elif sumber.startswith("Uji Sampel Test Set"):
            if os.path.exists("sample_test.csv"):
                df_input = pd.read_csv("sample_test.csv")
                nama_sumber = "sample_test.csv"
            else:
                st.error("File sample_test.csv tidak ditemukan.")

        elif sumber.startswith("Upload File CSV"):
            up_csv = st.file_uploader("Pilih file CSV yang memiliki 44 fitur statistik", type=["csv"])
            if up_csv is not None:
                df_input = pd.read_csv(up_csv)
                nama_sumber = up_csv.name

        elif sumber.startswith("Upload File PCAP"):
            up_pcap = st.file_uploader("Pilih file tangkapan jaringan (.pcap / .pcapng)", type=["pcap", "pcapng"])
            if up_pcap is not None:
                from predict_pcap import ekstrak
                with st.spinner("Mengekstrak 44 fitur per window 50 paket dari PCAP..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
                        tmp.write(up_pcap.read())
                        tmp_path = tmp.name
                    try:
                        rows, n_pkts = ekstrak(tmp_path, window_size=50, max_pkts=int(max_pkts_pcap))
                        if rows:
                            df_input = pd.DataFrame(rows)
                            if "Label" in df_input.columns:
                                df_input = df_input.drop(columns=["Label"])
                            nama_sumber = f"{up_pcap.name} ({n_pkts:,} paket IP)"
                        else:
                            st.warning("Tidak ada window yang terbentuk.")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

        if df_input is not None:
            kurang = [f for f in fitur_44 if f not in df_input.columns]
            if kurang:
                st.error(f"Kolom fitur kurang pada data: {kurang}")
            else:
                x_mat = df_input[fitur_44].astype(np.float32).values
                skor = booster.predict(x_mat)
                df_hasil = df_input.copy()
                df_hasil["skor_serangan"] = np.round(skor, 6)
                df_hasil["prediksi"] = np.where(skor >= threshold, "SERANGAN", "BENIGN")

                st.session_state["df_hasil"] = df_hasil
                st.session_state["nama_sumber"] = nama_sumber

                total_w = len(df_hasil)
                n_serangan = int((df_hasil["prediksi"] == "SERANGAN").sum())
                n_benign = total_w - n_serangan
                persen_atk = (n_serangan / total_w * 100.0) if total_w > 0 else 0.0

                st.markdown(f"**Sumber Data:** `{nama_sumber}` | **Threshold Aktif:** `{threshold:.2f}`")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Window", f"{total_w:,}")
                m2.metric("Prediksi SERANGAN", f"{n_serangan:,}", f"{persen_atk:.2f}% dari total", delta_color="inverse")
                m3.metric("Prediksi BENIGN", f"{n_benign:,}", f"{100 - persen_atk:.2f}% dari total")

                # Evaluasi Akurasi / Recall / FPR
                if "Label" in df_hasil.columns:
                    y_true = (df_hasil["Label"] != "Benign").astype(np.int32)
                    y_pred = (skor >= threshold).astype(np.int32)
                    tp = int(((y_pred == 1) & (y_true == 1)).sum())
                    fn = int(((y_pred == 0) & (y_true == 1)).sum())
                    fp = int(((y_pred == 1) & (y_true == 0)).sum())
                    tn = int(((y_pred == 0) & (y_true == 0)).sum())
                    rec_val = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
                    fpr_val = fp / (fp + tn) * 100.0 if (fp + tn) > 0 else 0.0
                    m4.metric("Recall / FPR (Label Asli)", f"Recall: {rec_val:.3f}%", f"FP Benign: {fp} ({fpr_val:.3f}%)")
                elif label_asli_pcap.startswith("Semua BENIGN"):
                    acc_benign = n_benign / total_w * 100.0 if total_w > 0 else 0.0
                    fpr_pcap = n_serangan / total_w * 100.0 if total_w > 0 else 0.0
                    m4.metric("Akurasi Benign (Uji FP)", f"{acc_benign:.2f}%", f"False Positive: {n_serangan} ({fpr_pcap:.2f}%)")
                elif label_asli_pcap.startswith("Semua SERANGAN"):
                    rec_pcap = n_serangan / total_w * 100.0 if total_w > 0 else 0.0
                    m4.metric("Recall Serangan", f"{rec_pcap:.2f}%", f"Lolos (FN): {n_benign} window")
                else:
                    m4.metric("Rata-rata Skor Model", f"{float(np.mean(skor)):.4f}")

                col_kiri, col_kanan = st.columns([3, 2])
                with col_kiri:
                    st.markdown("#### Tabel Hasil Prediksi per Window")
                    filter_status = st.selectbox(
                        "Filter Baris:",
                        ["Semua Window", "Hanya SERANGAN", "Hanya BENIGN"],
                    )
                    df_view = df_hasil
                    if filter_status == "Hanya SERANGAN":
                        df_view = df_hasil[df_hasil["prediksi"] == "SERANGAN"]
                    elif filter_status == "Hanya BENIGN":
                        df_view = df_hasil[df_hasil["prediksi"] == "BENIGN"]

                    kolom_utama = ["prediksi", "skor_serangan"]
                    if "pasangan_ip" in df_view.columns:
                        kolom_utama.append("pasangan_ip")
                    if "Label" in df_view.columns:
                        kolom_utama.append("Label")
                    kolom_sisa = [c for c in df_view.columns if c not in kolom_utama]
                    st.dataframe(df_view[kolom_utama + kolom_sisa].head(1000), use_container_width=True, height=340)

                    csv_bytes = df_hasil.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Unduh Hasil Prediksi (.CSV)",
                        data=csv_bytes,
                        file_name="hasil_prediksi_ddos.csv",
                        mime="text/csv",
                    )

                with col_kanan:
                    if "pasangan_ip" in df_hasil.columns:
                        st.markdown("#### Ringkasan per Pasangan IP")
                        df_atk_ip = df_hasil[df_hasil["prediksi"] == "SERANGAN"]
                        if len(df_atk_ip) > 0:
                            top_ip = (
                                df_atk_ip.groupby("pasangan_ip")
                                .agg(
                                    Window_Serangan=("prediksi", "count"),
                                    Skor_Maks=("skor_serangan", "max"),
                                    Rate_Rata2=("Rate", "mean"),
                                )
                                .sort_values("Window_Serangan", ascending=False)
                                .head(10)
                            )
                            st.dataframe(top_ip, use_container_width=True)
                        else:
                            st.info("Seluruh pasangan IP diklasifikasikan sebagai BENIGN pada threshold ini.")

                    if "Label" in df_hasil.columns:
                        st.markdown("#### Tabel Evaluasi Prediksi vs Label Asli")
                        ct = pd.crosstab(df_hasil["Label"], df_hasil["prediksi"], margins=True)
                        st.dataframe(ct, use_container_width=True)

                    st.markdown("#### Distribusi Skor Probabilitas")
                    hist_vals, hist_edges = np.histogram(skor, bins=10, range=(0.0, 1.0))
                    df_hist = pd.DataFrame({
                        "Rentang Skor": [f"{hist_edges[i]:.1f}-{hist_edges[i+1]:.1f}" for i in range(len(hist_vals))],
                        "Jumlah Window": hist_vals,
                    }).set_index("Rentang Skor")
                    st.bar_chart(df_hist)

    # =========================================================================
    # TAB 2: EXPLAINABLE AI (TREESHAP & COUNTERFACTUAL)
    # =========================================================================
    with tab_xai:
        st.subheader("Analisis Kontribusi Fitur (TreeSHAP) & Simulasi Nilai Fitur")
        col_g1, col_g2 = st.columns([1, 1])

        with col_g1:
            st.markdown("#### 12 Fitur Terpenting Global (Berdasarkan Gain)")
            imp_gain = booster.feature_importance(importance_type="gain")
            imp_split = booster.feature_importance(importance_type="split")
            total_gain = imp_gain.sum() if imp_gain.sum() > 0 else 1.0
            df_imp = pd.DataFrame({
                "Fitur": fitur_44,
                "Gain (%)": np.round(imp_gain / total_gain * 100.0, 3),
                "Jumlah Split": imp_split,
            }).sort_values("Gain (%)", ascending=False).reset_index(drop=True)
            st.dataframe(df_imp.head(12), use_container_width=True, height=350)

        with col_g2:
            st.markdown("#### Grafik Proporsi Gain Fitur")
            df_chart_imp = df_imp.head(10).set_index("Fitur")[["Gain (%)"]]
            st.bar_chart(df_chart_imp)

        st.markdown("---")
        st.markdown("#### Bedah Kontribusi Fitur per Window (TreeSHAP) & Uji Simulasi")
        df_aktif = st.session_state.get("df_hasil", None)
        if df_aktif is None:
            st.info("Silakan muat data terlebih dahulu di Tab 1.")
        else:
            pilihan_idx = st.number_input(
                f"Pilih Indeks Baris Window (0 s/d {len(df_aktif) - 1}) dari {st.session_state.get('nama_sumber', '')}:",
                min_value=0,
                max_value=max(len(df_aktif) - 1, 0),
                value=int(df_aktif.index[df_aktif["prediksi"] == "SERANGAN"][0]) if (df_aktif["prediksi"] == "SERANGAN").any() else 0,
                step=1,
            )
            baris = df_aktif.iloc[int(pilihan_idx)]
            x_row = baris[fitur_44].astype(np.float32).values.reshape(1, -1)
            contrib = booster.predict(x_row, pred_contrib=True)[0]
            bias_val = float(contrib[-1])
            shap_vals = contrib[:-1]

            df_shap = pd.DataFrame({
                "Fitur": fitur_44,
                "Nilai Fitur": [float(baris[f]) for f in fitur_44],
                "Kontribusi Log-Odds": np.round(shap_vals, 4),
                "Keterangan": ["Mendorong ke SERANGAN (+)" if v > 0 else "Mendorong ke BENIGN (-)" for v in shap_vals],
            }).reindex(pd.Series(np.abs(shap_vals)).sort_values(ascending=False).index).reset_index(drop=True)

            k1, k2 = st.columns([3, 2])
            with k1:
                st.markdown(
                    f"**Window #{pilihan_idx}** | Prediksi: **{baris['prediksi']}** | "
                    f"Skor Asli: **{baris['skor_serangan']:.4f}** | "
                    f"Bias Dasar: `{bias_val:.3f}` | Total Kontribusi Fitur: `{shap_vals.sum():.3f}`"
                )
                st.dataframe(df_shap.head(12), use_container_width=True, height=340)

            with k2:
                st.markdown("##### Simulasi Perubahan Nilai Fitur (Counterfactual)")
                st.caption("Ubah nilai fitur di bawah ini untuk melihat pengaruhnya terhadap skor prediksi model.")
                sim_srate = st.number_input("Nilai Srate (paket keluar/detik):", value=float(baris["Srate"]), format="%.4f")
                sim_rate = st.number_input("Nilai Rate (total paket/detik):", value=float(baris["Rate"]), format="%.4f")
                sim_min = st.number_input("Nilai Min (ukuran paket minimum byte):", value=float(baris["Min"]), format="%.1f")
                sim_dur = st.number_input("Nilai Duration (rata-rata TTL):", value=float(baris["Duration"]), format="%.2f")

                x_sim = baris[fitur_44].astype(np.float32).copy()
                x_sim["Srate"] = sim_srate
                x_sim["Rate"] = sim_rate
                x_sim["Min"] = sim_min
                x_sim["Duration"] = sim_dur
                skor_sim = float(booster.predict(x_sim.values.reshape(1, -1))[0])
                status_sim = "SERANGAN" if skor_sim >= threshold else "BENIGN"

                st.metric(
                    "Skor Prediksi Simulasi",
                    f"{skor_sim:.4f} ({status_sim})",
                    delta=f"{skor_sim - float(baris['skor_serangan']):+.4f} dari skor asli",
                    delta_color="inverse",
                )

    # =========================================================================
    # TAB 3: HASIL EVALUASI & AUDIT DATASET
    # =========================================================================
    with tab_evaluasi:
        st.subheader("Evaluasi Resmi pada Dataset CIC-DDoS2019 (Test Set Temporal)")
        st.markdown(
            """
            Data dibagi berdasarkan urutan waktu per kelas: 70% awal untuk pelatihan, 15% tengah untuk validasi, dan 15% akhir untuk pengujian.
            Seluruh baris telah melalui proses deduplikasi hash SHA-256 sehingga tidak ada kebocoran data antar-split.
            """
        )

        e1, e2 = st.columns([3, 2])
        with e1:
            st.markdown("#### Deteksi per Jenis Serangan (Test Set: 75.920 Baris)")
            df_bench = pd.DataFrame(BENCHMARK_PER_KELAS)
            st.dataframe(df_bench, use_container_width=True, height=420)

        with e2:
            st.markdown("#### Ringkasan Audit Data (clean_dedup.py)")
            audit = load_audit_summary()
            if audit:
                st.json({
                    "Mode Split": audit["split_mode"],
                    "Total Baris Mentah": audit["raw_total"],
                    "Baris NaN/Inf Dibuang": audit["bad_rows"],
                    "Filter Rate Minimum Serangan": audit["min_rate_serangan"],
                    "Duplikat SHA-256 Dibuang": audit["duplicate_rows"],
                    "Total Baris Unik Bersih": audit["unique_rows"],
                    "Cek Kebocoran Antar-Split": audit["leakage"],
                })
            st.info(
                "Catatan Evaluasi:\n"
                "Recall mencapai 100% pada 13 dari 14 jenis serangan. "
                "Sebanyak 12 baris yang lolos pada kelas TFTP (Recall 99,91%) berada di ujung akhir segmen waktu serangan saat laju paket menurun."
            )


if __name__ == "__main__":
    main()
