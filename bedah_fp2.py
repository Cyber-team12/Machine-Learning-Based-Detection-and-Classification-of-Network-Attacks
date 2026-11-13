import json
import numpy as np
import pandas as pd
import lightgbm as lgb

f = json.load(open("fitur_terpilih_44.json", encoding="utf-8"))
d = pd.read_csv("hasil_pcap.csv")
b = lgb.Booster(model_file="model_lgb_final_44fitur.txt")
x = d.loc[d.prediksi == "SERANGAN", f].astype(np.float32)

c = b.predict(x.values, pred_contrib=True)
print("nilai dasar model (bias):", round(float(c[:, -1].mean()), 3))
print("total kontribusi semua fitur:", round(float(c[:, :-1].sum(axis=1).mean()), 3))
print("skor asli:", b.predict(x.values).round(3))
x2 = x.copy()
x2["Min"] = 54
print("skor kalau Min diganti 54:", b.predict(x2.values).round(3))