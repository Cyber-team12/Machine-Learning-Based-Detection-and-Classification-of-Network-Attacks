import collections
import csv
from datetime import datetime
import hashlib
import json
import os
import lightgbm as lgb
import numpy as np

TRAIN_CSV = "csv_bersih/dataset_clean_train.csv"
VAL_CSV = "csv_bersih/dataset_clean_val.csv"
ARTEFAK_MODEL = "model_lgb_final_44fitur.txt"
FITUR_JSON = "fitur_terpilih_44.json"

SEED = 42
THRESHOLD_OPERASIONAL = 0.5
SIKLUS_ITERASI = [35, 35, 35]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "max_depth": 8,
    "min_data_in_leaf": 20,
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

def main():
    print(f"Membaca {TRAIN_CSV} dan {VAL_CSV}")
    x_train, y_train, _, fitur = baca_csv(TRAIN_CSV)
    x_val, y_val, label_val, _ = baca_csv(VAL_CSV)
    print(f"Baris train: {len(y_train):,} | Baris val: {len(y_val):,} | Fitur: {len(fitur)}")

    with open(FITUR_JSON, "w", encoding="utf-8") as jf:
        json.dump(fitur, jf, indent=2)
    bobot = np.where(y_train == 0, 5.0, 1.0).astype(np.float32)

    model = None
    for tahap, n_iter in enumerate(SIKLUS_ITERASI, start=1):
        dtrain = lgb.Dataset(x_train, label=y_train, weight=bobot, feature_name=fitur, free_raw_data=False)
        dval = lgb.Dataset(x_val, label=y_val, reference=dtrain, feature_name=fitur, free_raw_data=False)
        print(f"\nTahap training ulang {tahap}/{len(SIKLUS_ITERASI)} (+{n_iter} iterasi):")
        model = lgb.train(
            params=LGB_PARAMS,
            train_set=dtrain,
            num_boost_round=n_iter,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            init_model=model,
            keep_training_booster=True,
            callbacks=[lgb.log_evaluation(period=15)],
        )

        prob_v = model.predict(x_val)
        _, fn_v, fp_v, _, rec_v, fpr_v, _ = evaluasi_ringkas(y_val, prob_v)
        print(
            f"Evaluasi tahap {tahap} (total {model.current_iteration()} pohon) -> "
            f"Recall: {rec_v * 100:.4f}% (FN: {fn_v:,}) | FP Benign: {fp_v:,} ({fpr_v * 100:.4f}%)"
        )
        # Koreksi bobot bertahap pada sampel yang meleset
        if tahap < len(SIKLUS_ITERASI):
            prob_tr = model.predict(x_train)
            pred_tr = (prob_tr >= THRESHOLD_OPERASIONAL).astype(np.int32)
            fn_mask = (pred_tr == 0) & (y_train == 1)
            fp_mask = (pred_tr == 1) & (y_train == 0)
            bobot[fn_mask] *= 1.5
            bobot[fp_mask] *= 1.8
            bobot = np.clip(bobot, 0.5, 20.0)
    path_model = cek_path_model(ARTEFAK_MODEL)
    model.save_model(path_model)
    sha_model = hitung_sha256(path_model)

    print(f"\nModel disimpan di: {path_model}")
    print(f"SHA-256 model: {sha_model}")
    print(f"Perintah uji test: python tes_resmi.py --model {path_model}")
    prob_val = model.predict(x_val)
    tp, fn, fp, tn, recall, fpr, pred_val = evaluasi_ringkas(y_val, prob_val)
    roc_auc, pr_auc = hitung_roc_pr_auc(y_val, prob_val)
    print(f"\nHasil validasi akhir ({len(y_val):,} baris, seed {SEED}):")
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