import argparse
import collections
import json
import os
import numpy as np
import lightgbm as lgb
from train import baca_csv, evaluasi_ringkas, hitung_roc_pr_auc

TEST_CSV = "csv_bersih/dataset_clean_test.csv"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_lgb_final_44fitur.txt")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--test-csv", default=TEST_CSV)
    args = ap.parse_args()

    th = args.threshold
    if th is None:
        tpath = args.model + ".threshold.json"
        th = json.load(open(tpath))["threshold"] if os.path.exists(tpath) else 0.5
    x, y, lbl, fitur = baca_csv(args.test_csv)
    booster = lgb.Booster(model_file=args.model)
    p = booster.predict(x)
    tp, fn, fp, tn, rec, fpr, pred = evaluasi_ringkas(y, p, th)
    roc, pr = hitung_roc_pr_auc(y, p)
    print(f"Model: {args.model} | threshold: {th:.2f} | baris test: {len(y):,}")
    print(f"Recall serangan: {rec * 100:.4f}% ({fn:,} lolos dari {tp + fn:,})")
    print(f"FP benign: {fp:,} dari {fp + tn:,} ({fpr * 100:.4f}%)")
    print(f"ROC-AUC: {roc:.7f} | PR-AUC: {pr:.7f}")

    print("\nPeka terhadap threshold:")
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        _, fn_t, fp_t, _, rec_t, fpr_t, _ = evaluasi_ringkas(y, p, t)
        print(f"  t={t:.1f} | recall={rec_t * 100:.4f}% (lolos {fn_t:,}) | FP benign={fp_t:,} ({fpr_t * 100:.3f}%)")
    tot = collections.Counter(lbl)
    lolos = collections.Counter(l for l, pp, yy in zip(lbl, pred, y) if yy == 1 and pp == 0)
    print("\nPer jenis serangan:")
    for k in sorted(tot):
        if k == "Benign":
            continue
        print(f"  {k:15s} | recall={(tot[k] - lolos[k]) / tot[k] * 100:7.3f}% | lolos={lolos[k]:5d} / {tot[k]:6d}")

    imp = booster.feature_importance(importance_type="gain")
    urut = np.argsort(imp)[::-1][:10]
    total = imp.sum() if imp.sum() > 0 else 1
    print("\n10 fitur terpenting (gain):")
    for i in urut:
        print(f"  {fitur[i]:18s} {imp[i] / total * 100:6.2f}%")

if __name__ == "__main__":
    main()