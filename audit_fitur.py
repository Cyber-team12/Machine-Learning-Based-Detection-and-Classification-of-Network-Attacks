import csv
import json
import os

TRAIN_CSV = "csv_bersih/dataset_clean_train.csv"
SUMMARY_JSON = "csv_bersih/audit_clean_summary.json"

def main():
    if os.path.exists(SUMMARY_JSON):
        with open(SUMMARY_JSON, "r", encoding="utf-8") as jf:
            summary = json.load(jf)
        print(f"Ringkasan clean_dedup:")
        print(f"  Total mentah   : {summary['raw_total']}")
        print(f"  Duplikat hapus : {summary['duplicate_rows']} (konflik label: {summary['label_conflicts']})")
        print(f"  Total unik     : {summary['unique_rows']}")
        print(f"  Leakage cek    : {summary['leakage']}\n")

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"File tidak ditemukan: {TRAIN_CSV}")
    with open(TRAIN_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        features = header[:-1]
        n_feat = len(features)
        min_val = [float("inf")] * n_feat
        max_val = [float("-inf")] * n_feat
        zero_count = [0] * n_feat
        benign_zero = [0] * n_feat
        attack_zero = [0] * n_feat
        n_benign = 0
        n_attack = 0
        total_rows = 0
        for row in reader:
            if not row:
                continue
            total_rows += 1
            is_benign = row[-1].strip() == "Benign"
            if is_benign:
                n_benign += 1
            else:
                n_attack += 1

            for i in range(n_feat):
                v = float(row[i])
                if v < min_val[i]:
                    min_val[i] = v
                if v > max_val[i]:
                    max_val[i] = v
                if v == 0.0:
                    zero_count[i] += 1
                    if is_benign:
                        benign_zero[i] += 1
                    else:
                        attack_zero[i] += 1
    constant_features = []
    shortcut_candidates = []

    for i, name in enumerate(features):
        if min_val[i] == max_val[i]:
            constant_features.append(name)
            continue

        # Cek apakah aturan sederhana
        if n_benign > 0 and n_attack > 0:
            b_zero_rate = benign_zero[i] / n_benign
            a_zero_rate = attack_zero[i] / n_attack
            gap = abs(b_zero_rate - a_zero_rate)
            if gap >= 0.90:
                shortcut_candidates.append((name, gap, b_zero_rate, a_zero_rate))

    print(f"Total train rows : {total_rows} (Benign: {n_benign}, Attack: {n_attack})")
    print(f"Fitur konstan: {len(constant_features)} -> {constant_features}")
    print("\nKandidat fitur pintasan:")
    if not shortcut_candidates:
        print("  Tidak ditemukan fitur pintasan ekstrem pada aturan == 0.")
    else:
        for name, gap, bz, az in sorted(shortcut_candidates, key=lambda x: x[1], reverse=True):
            print(f"  - {name:18s} | gap={gap:.4f} | benign_zero={bz:.4f} | attack_zero={az:.4f}")


if __name__ == "__main__":
    main()