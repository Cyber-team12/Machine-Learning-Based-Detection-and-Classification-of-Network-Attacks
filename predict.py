import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

MODEL = "model_lgb_final_44fitur.txt"
FITUR_JSON = "fitur_terpilih_44.json"

def main():
    ap = argparse.ArgumentParser(description="Deteksi DDoS dari CSV fitur (44 fitur)")
    ap.add_argument("csv", help="CSV berisi 44 fitur (kolom Label opsional)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default="hasil_prediksi.csv")
    args = ap.parse_args()

    th = args.threshold
    if th is None:
        tpath = args.model + ".threshold.json"
        th = json.load(open(tpath))["threshold"] if os.path.exists(tpath) else 0.5
    fitur = json.load(open(FITUR_JSON, encoding="utf-8"))
    df = pd.read_csv(args.csv)
    kurang = [f for f in fitur if f not in df.columns]
    if kurang:
        sys.exit(f"Kolom fitur kurang: {kurang}")

    x = df[fitur].astype(np.float32).values
    p = lgb.Booster(model_file=args.model).predict(x)
    df["skor_serangan"] = p
    df["prediksi"] = np.where(p >= th, "SERANGAN", "BENIGN")
    df.to_csv(args.out, index=False)
    n_atk = int((p >= th).sum())
    print(f"Threshold: {th}")
    print(f"Total baris: {len(df):,} | SERANGAN: {n_atk:,} | BENIGN: {len(df) - n_atk:,}")
    if "Label" in df.columns:
        print("\nPrediksi vs label asli:")
        print(pd.crosstab(df["Label"], df["prediksi"]))
    print(f"\nHasil disimpan di: {args.out}")

if __name__ == "__main__":
    main()