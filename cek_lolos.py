import numpy as np
import lightgbm as lgb
from train import baca_csv

MODEL = "model_lgb_final_44fitur.txt"
TEST_CSV = "csv_bersih/dataset_clean_test.csv"
TRAIN_CSV = "csv_bersih/dataset_clean_train.csv"
THRESHOLD = 0.5
KELAS = ["Syn", "TFTP", "MSSQL", "DrDoS_NTP"]
FITUR = ["Srate", "Rate", "AVG", "flow_duration"]

def main():
    x, y, lbl, fitur = baca_csv(TEST_CSV)
    lbl = np.array(lbl)
    p = lgb.Booster(model_file=MODEL).predict(x)
    idx = [fitur.index(f) for f in FITUR]
    xt, yt, lt, _ = baca_csv(TRAIN_CSV)
    lt = np.array(lt)
    print("Median fitur Benign (train) sebagai pembanding:")
    print("  " + " | ".join(f"{f}={np.median(xt[lt == 'Benign'][:, fitur.index(f)]):.4g}" for f in FITUR))
    print()

    for k in KELAS:
        pos = np.where(lbl == k)[0]
        n = len(pos)
        lolos_mask = p[pos] < THRESHOLD
        n_lolos = int(lolos_mask.sum())
        print(f"[{k}] total test={n} | lolos={n_lolos}")
        if n_lolos == 0:
            print()
            continue
        frac = np.where(lolos_mask)[0] / n 
        print(f"posisi lolos dalam urutan waktu: min={frac.min():.2f} median={np.median(frac):.2f} max={frac.max():.2f}")
        print(f"skor model baris lolos: median={np.median(p[pos][lolos_mask]):.3f}")
        for nama, m in [("lolos", lolos_mask), ("terdeteksi", ~lolos_mask)]:
            if m.sum() == 0:
                continue
            sub = x[pos][m][:, idx]
            print(f"  median fitur {nama:10s}: " + " | ".join(f"{f}={np.median(sub[:, i]):.4g}" for i, f in enumerate(FITUR)))
        print()

if __name__ == "__main__":
    main()