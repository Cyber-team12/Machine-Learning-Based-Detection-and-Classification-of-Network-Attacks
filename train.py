import argparse
import collections
import csv
from datetime import datetime
import hashlib
import json
import math
import os
import lightgbm as lgb
import numpy as np

TRAIN_CSV = "csv_bersih/dataset_clean_train.csv"
VAL_CSV = "csv_bersih/dataset_clean_val.csv"
ARTEFAK_MODEL = "model_lgb_final_44fitur.txt"
FITUR_JSON = "fitur_terpilih_44.json"
SEED = 42
THRESHOLD_OPERASIONAL = 0.5

# Early stopping level tahap
ITER_PER_TAHAP = 100  # pohon yang ditambah tiap tahap
MAX_TAHAP = 8         # batas atas jumlah tahap
PATIENCE = 2          # stop kalau N tahap berturut-turut tidak membaik
MIN_DELTA = 0.001     # minimal turun FNR+FPR segini baru dianggap membaik

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "max_depth": 10,
    "min_data_in_leaf": 15,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "n_jobs": -1,
    "verbose": -1,
    "seed": SEED,
}
def cek_path_model(path_utama):
    if not os.path.exists(path_utama):
        return path_utama
    nama, ekstensi = os.path.splitext(path_utama)
    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_baru = f"{nama}_v_{waktu}{ekstensi}"
    print(f"File {path_utama} sudah ada, model baru disimpan ke {path_baru}")
    return path_baru
def hitung_sha256(path_file):
    h = hashlib.sha256()
    with open(path_file, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
def baca_csv(path_csv):
    with open(path_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        fitur = header[:-1]
        x_list, y_list, label_asli = [], [], []
        for row in reader:
            if not row:
                continue
            x_list.append([float(v) for v in row[:-1]])
            lbl = row[-1].strip()
            label_asli.append(lbl)
            y_list.append(0 if lbl == "Benign" else 1)
    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.int32), label_asli, fitur

def hitung_roc_pr_auc(y_true, y_score):
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    total_pos = tps[-1] if len(tps) > 0 else 1
    total_neg = fps[-1] if len(fps) > 0 else 1
    tpr = np.concatenate([[0.0], tps / total_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / total_neg, [1.0]])
    roc_auc = float(np.trapezoid(tpr, fpr))
    precision = tps / (tps + fps)
    recall = tps / total_pos
    pr_auc = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]) + recall[0] * precision[0])
    return roc_auc, pr_auc

def evaluasi_ringkas(y_true, prob, threshold=THRESHOLD_OPERASIONAL):
    pred = (prob >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tp, fn, fp, tn, recall, fpr, pred

def hitung_bobot_subtipe(label_train):
    counts = collections.Counter(label_train)
    attack_counts = [c for lbl, c in counts.items() if lbl != "Benign"]
    mean_attack = sum(attack_counts) / max(len(attack_counts), 1)
    w_map = {}
    for lbl, c in counts.items():
        if lbl == "Benign":
            w_map[lbl] = max(mean_attack / max(c, 1) * 1.5, 3.0)
        else:
            w_map[lbl] = math.sqrt(mean_attack / max(c, 1))
    return np.array([w_map[lbl] for lbl in label_train], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="csv_bersih")
    ap.add_argument("--model-out", default=ARTEFAK_MODEL)
    args = ap.parse_args()
    train_csv = os.path.join(args.data_dir, "dataset_clean_train.csv")
    val_csv = os.path.join(args.data_dir, "dataset_clean_val.csv")
    print(f"Membaca {train_csv} dan {val_csv}")
    x_train, y_train, label_train, fitur = baca_csv(train_csv)
    x_val, y_val, label_val, _ = baca_csv(val_csv)
    print(f"Baris train: {len(y_train):,} | Baris val: {len(y_val):,} | Fitur: {len(fitur)}")
    with open(FITUR_JSON, "w", encoding="utf-8") as jf:
        json.dump(fitur, jf, indent=2)
    bobot = hitung_bobot_subtipe(label_train)
    bobot_max = float(bobot.max()) * 3.0 
    n_pos = int((y_val == 1).sum())
    n_neg = int((y_val == 0).sum())
    model = None
    best_model_str = None
    best_err = float("inf")
    best_tahap = 0
    tanpa_perbaikan = 0

    for tahap in range(1, MAX_TAHAP + 1):
        dtrain = lgb.Dataset(x_train, label=y_train, weight=bobot, feature_name=fitur, free_raw_data=False)
        dval = lgb.Dataset(x_val, label=y_val, reference=dtrain, feature_name=fitur, free_raw_data=False)
        print(f"\nTahap {tahap}/{MAX_TAHAP} (+{ITER_PER_TAHAP} iterasi):")
        model = lgb.train(
            params=LGB_PARAMS,
            train_set=dtrain,
            num_boost_round=ITER_PER_TAHAP,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            init_model=model,
            keep_training_booster=True,
            callbacks=[lgb.log_evaluation(period=25)],
        )
        prob_v = model.predict(x_val)
        _, fn_v, fp_v, _, rec_v, fpr_v, _ = evaluasi_ringkas(y_val, prob_v)
        err = fn_v / n_pos + fp_v / n_neg
        print(
            f"Evaluasi tahap {tahap} (total {model.current_iteration()} pohon) -> "
            f"Recall: {rec_v * 100:.4f}% (FN: {fn_v:,}) | FP Benign: {fp_v:,} ({fpr_v * 100:.4f}%) | FNR+FPR: {err:.4f}"
        )

        if err <= best_err - MIN_DELTA:
            best_err = err
            best_tahap = tahap
            best_model_str = model.model_to_string()
            tanpa_perbaikan = 0
            print("membaik, model terbaik diperbarui")
        else:
            tanpa_perbaikan += 1
            print(f"tidak membaik ({tanpa_perbaikan}/{PATIENCE})")
            if tanpa_perbaikan >= PATIENCE:
                print(f"Early stop: tidak ada perkembangan, pakai model tahap {best_tahap}")
                break
        prob_tr = model.predict(x_train)
        pred_tr = (prob_tr >= THRESHOLD_OPERASIONAL).astype(np.int32)
        fn_mask = (pred_tr == 0) & (y_train == 1)
        fp_mask = (pred_tr == 1) & (y_train == 0)
        bobot[fn_mask] *= 1.2
        bobot[fp_mask] *= 1.4
        bobot = np.clip(bobot, 0.2, bobot_max)

    # Balik ke model terbaik
    model = lgb.Booster(model_str=best_model_str)
    print(f"\nModel terbaik dari tahap {best_tahap} (FNR+FPR val: {best_err:.4f})")
    path_model = cek_path_model(args.model_out)
    model.save_model(path_model)
    sha_model = hitung_sha256(path_model)
    print(f"\nModel disimpan di: {path_model}")
    print(f"SHA-256 model: {sha_model}")
    print(f"Perintah uji test: python tes_resmi.py --model {path_model}")
    prob_val = model.predict(x_val)

    # Sweep threshold
    print("\nSweep threshold (val):")
    hasil_sweep = []
    for t in np.arange(0.05, 0.96, 0.05):
        _, fn_t, fp_t, _, rec_t, fpr_t, _ = evaluasi_ringkas(y_val, prob_val, t)
        sk = fn_t / n_pos + fp_t / n_neg
        print(f"  t={t:.2f} | recall={rec_t * 100:.3f}% (FN {fn_t:,}) | FPR={fpr_t * 100:.2f}% (FP {fp_t:,})")
        hasil_sweep.append((sk, float(t)))
    min_sk = min(sk for sk, _ in hasil_sweep)
    # Kalau banyak threshold seri
    kandidat = sorted(t for sk, t in hasil_sweep if sk <= min_sk + 1e-9)
    best_t = kandidat[len(kandidat) // 2]
    print(f"Threshold terbaik (FNR+FPR minimum, seri diambil tengah): {best_t:.2f}")
    with open(path_model + ".threshold.json", "w", encoding="utf-8") as tf:
        json.dump({"threshold": best_t}, tf)
    tp, fn, fp, tn, recall, fpr, pred_val = evaluasi_ringkas(y_val, prob_val, best_t)
    roc_auc, pr_auc = hitung_roc_pr_auc(y_val, prob_val)

    print(f"\nHasil validasi akhir (threshold {best_t:.2f}, {len(y_val):,} baris, seed {SEED}):")
    print(f"Recall serangan: {recall * 100:.4f}% ({fn:,} lolos dari {tp + fn:,})")
    print(f"FP benign: {fp:,} dari {fp + tn:,} ({fpr * 100:.4f}%)")
    print(f"ROC-AUC: {roc_auc:.7f} | PR-AUC: {pr_auc:.7f}")
    total_per_jenis = collections.Counter(label_val)
    lolos_per_jenis = collections.Counter()
    for lbl, p, y in zip(label_val, pred_val, y_val):
        if y == 1 and p == 0:
            lolos_per_jenis[lbl] += 1

    print("\nRincian deteksi per jenis serangan:")
    for lbl in sorted(total_per_jenis.keys()):
        if lbl == "Benign":
            continue
        tot = total_per_jenis[lbl]
        lolos = lolos_per_jenis[lbl]
        rec_lbl = (tot - lolos) / tot * 100.0
        print(f"  {lbl:15s} | recall={rec_lbl:7.3f}% | lolos={lolos:5d} / {tot:6d}")

if __name__ == "__main__":
    main()