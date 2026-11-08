import collections
import csv
import hashlib
import json
import os
import random

INPUT_CSV = "csv_mentah/dataset_44fitur_raw.csv"
OUT_DIR = "csv_bersih"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42
SPLIT_MODE = "waktu"
MIN_RATE_SERANGAN = 1000.0
INVALID_VALUES = {"", "nan", "inf", "-inf", "infinity", "-infinity"}

def row_hash(features):
    return hashlib.sha256(",".join(features).encode("utf-8")).digest()

def is_invalid(features):
    return any(v.strip().lower() in INVALID_VALUES for v in features)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_CSV}")
    if SPLIT_MODE not in ("waktu", "acak"):
        raise ValueError("SPLIT_MODE harus 'waktu' atau 'acak'")

    os.makedirs(OUT_DIR, exist_ok=True)
    raw_total = 0
    bad_rows = 0
    dup_rows = 0
    label_conflicts = 0
    dibuang_rate_rendah = collections.Counter()
    seen_hashes = {}
    rows_by_class = collections.defaultdict(list)
    with open(INPUT_CSV, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rate_idx = header.index("Rate")
        for row in reader:
            if not row or len(row) != len(header):
                continue
            raw_total += 1
            features = row[:-1]
            label = row[-1].strip()
            if is_invalid(features):
                bad_rows += 1
                continue
            if MIN_RATE_SERANGAN > 0 and label != "Benign" and float(features[rate_idx]) < MIN_RATE_SERANGAN:
                dibuang_rate_rendah[label] += 1
                continue
            h = row_hash(features)
            if h in seen_hashes:
                dup_rows += 1
                if seen_hashes[h] != label:
                    label_conflicts += 1
                continue
            seen_hashes[h] = label
            rows_by_class[label].append((h, features + [label]))

    print(f"Mode split: {SPLIT_MODE}")
    print(f"Total baris: {raw_total}")
    print(f"NaN/Inf dibuang: {bad_rows}")
    print(f"Serangan ber-Rate < {MIN_RATE_SERANGAN:g} dibuang: {sum(dibuang_rate_rendah.values())}")
    for k, v in sorted(dibuang_rate_rendah.items(), key=lambda kv: -kv[1]):
        print(f"    {k:15s} {v}")
    print(f"Duplikat dibuang: {dup_rows} (konflik label: {label_conflicts})")
    print(f"Total unik: {len(seen_hashes)}")
    rng = random.Random(SEED)
    train_rows, val_rows, test_rows = [], [], []
    train_hashes, val_hashes, test_hashes = set(), set(), set()
    split_summary = {}

    for label, items in sorted(rows_by_class.items()):
        items_copy = list(items)
        if SPLIT_MODE == "acak":
            rng.shuffle(items_copy)
        n = len(items_copy)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        s_train = items_copy[:n_train]
        s_val = items_copy[n_train:n_train + n_val]
        s_test = items_copy[n_train + n_val:]
        for h, r in s_train:
            train_rows.append(r)
            train_hashes.add(h)
        for h, r in s_val:
            val_rows.append(r)
            val_hashes.add(h)
        for h, r in s_test:
            test_rows.append(r)
            test_hashes.add(h)
        split_summary[label] = {"total": n, "train": len(s_train), "val": len(s_val), "test": len(s_test)}
        print(f"{label:15s} | total={n:7d} | train={len(s_train):7d} | val={len(s_val):6d} | test={len(s_test):6d}")
    leak_tv = len(train_hashes & val_hashes)
    leak_tt = len(train_hashes & test_hashes)
    leak_vt = len(val_hashes & test_hashes)
    print(f"Cek konflik hash -> train-val: {leak_tv}, train-test: {leak_tt}, val-test: {leak_vt}")

    for fname, data in [
        ("dataset_clean_train.csv", train_rows),
        ("dataset_clean_val.csv", val_rows),
        ("dataset_clean_test.csv", test_rows),
    ]:
        path = os.path.join(OUT_DIR, fname)
        with open(path, mode="w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)
            writer.writerows(data)
        print(f"Disimpan: {path} ({len(data)} baris)")

    summary = {
        "split_mode": SPLIT_MODE,
        "raw_total": raw_total,
        "bad_rows": bad_rows,
        "min_rate_serangan": MIN_RATE_SERANGAN,
        "dibuang_rate_rendah": dict(dibuang_rate_rendah),
        "duplicate_rows": dup_rows,
        "label_conflicts": label_conflicts,
        "unique_rows": len(seen_hashes),
        "splits": split_summary,
        "leakage": {"train_val": leak_tv, "train_test": leak_tt, "val_test": leak_vt},
    }
    with open(os.path.join(OUT_DIR, "audit_clean_summary.json"), "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

if __name__ == "__main__":
    main()