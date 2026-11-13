import json
import numpy as np
import pandas as pd
import lightgbm as lgb

f = json.load(open("fitur_terpilih_44.json", encoding="utf-8"))
d = pd.read_csv("hasil_pcap.csv")
b = lgb.Booster(model_file="model_lgb_final_44fitur.txt")
c = b.predict(d[f].astype(np.float32).values, pred_contrib=True)[:, :-1]
idx = d.index[d.prediksi == "SERANGAN"]
rata = pd.Series(c[idx].mean(axis=0), index=f)
print("Kontribusi rata-rata ke 6 window yang kena (positif = dorong ke SERANGAN):")
print(rata.sort_values(key=abs, ascending=False).head(8).to_string())