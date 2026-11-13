import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scapy.all import PcapReader, IP, TCP, UDP, ICMP
from convert_csv import compute_44_features

MODEL = "model_lgb_final_44fitur.txt"
FITUR_JSON = "fitur_terpilih_44.json"

def ekstrak(pcap_path, window_size=50, max_pkts=200000):
    buffers, rows, n = {}, [], 0
    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            if IP not in pkt:
                continue
            n += 1
            ts = float(pkt.time)
            src, dst = pkt[IP].src, pkt[IP].dst
            hdr_len = pkt[IP].ihl * 4 if getattr(pkt[IP], "ihl", None) else 20
            sport = dport = 0
            fin = syn = rst = psh = ack = ece = cwr = 0
            is_tcp = is_udp = is_icmp = 0
            if TCP in pkt:
                is_tcp = 1
                sport, dport = pkt[TCP].sport, pkt[TCP].dport
                hdr_len += pkt[TCP].dataofs * 4 if pkt[TCP].dataofs else 20
                fl = str(pkt[TCP].flags)
                fin, syn, rst, psh, ack, ece, cwr = (
                    int("F" in fl), int("S" in fl), int("R" in fl),
                    int("P" in fl), int("A" in fl), int("E" in fl), int("C" in fl))
            elif UDP in pkt:
                is_udp = 1
                sport, dport = pkt[UDP].sport, pkt[UDP].dport
                hdr_len += 8
            elif ICMP in pkt:
                is_icmp = 1
                hdr_len += 8
            ports = {sport, dport}
            key = tuple(sorted([src, dst]))
            buffers.setdefault(key, []).append({
                "ts": ts, "size": len(pkt), "hdr_len": hdr_len, "ttl": pkt[IP].ttl,
                "proto": pkt[IP].proto, "dir": 1 if src <= dst else 0,
                "fin": fin, "syn": syn, "rst": rst, "psh": psh, "ack": ack,
                "ece": ece, "cwr": cwr, "tcp": is_tcp, "udp": is_udp, "icmp": is_icmp,
                "http": int(80 in ports or 8080 in ports), "https": int(443 in ports),
                "dns": int(53 in ports), "telnet": int(23 in ports), "smtp": int(25 in ports),
                "ssh": int(22 in ports), "irc": int(194 in ports or 6667 in ports),
                "dhcp": int(67 in ports or 68 in ports)})
            if len(buffers[key]) >= window_size:
                r = compute_44_features(buffers[key], "?")
                if r:
                    r["pasangan_ip"] = f"{key[0]} <-> {key[1]}"
                    rows.append(r)
                buffers[key] = []
            if n >= max_pkts:
                break
    for key, left in buffers.items():
        if len(left) >= 5:
            r = compute_44_features(left, "?")
            if r:
                r["pasangan_ip"] = f"{key[0]} <-> {key[1]}"
                rows.append(r)
    return rows, n

def main():
    ap = argparse.ArgumentParser(description="Deteksi DDoS langsung dari file PCAP")
    ap.add_argument("pcap")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default="hasil_pcap.csv")
    ap.add_argument("--max-pkts", type=int, default=200000)
    ap.add_argument("--label-asli", choices=["benign", "serangan"], default="benign",
                    help="Asumsi label asli seluruh trafik PCAP untuk menghitung Akurasi/FPR/Recall (default: benign)")
    a = ap.parse_args()
    th = a.threshold
    if th is None:
        tp = a.model + ".threshold.json"
        th = json.load(open(tp))["threshold"] if os.path.exists(tp) else 0.5

    rows, n = ekstrak(a.pcap, max_pkts=a.max_pkts)
    if not rows:
        sys.exit("Tidak ada window yang terbentuk (PCAP kosong / tidak ada paket IP).")
    fitur = json.load(open(FITUR_JSON, encoding="utf-8"))
    df = pd.DataFrame(rows)
    p = lgb.Booster(model_file=a.model).predict(df[fitur].astype(np.float32).values)
    df["skor_serangan"] = p
    df["prediksi"] = np.where(p >= th, "SERANGAN", "BENIGN")
    df.drop(columns=["Label"]).to_csv(a.out, index=False)
    total_w = len(df)
    n_atk = int((p >= th).sum())
    n_ben = total_w - n_atk
    print(f"Paket IP dibaca: {n:,} | window: {total_w:,} | threshold: {th}")
    print(f"SERANGAN: {n_atk:,} | BENIGN: {n_ben:,}")
    if a.label_asli == "benign":
        tnr = n_ben / total_w * 100.0
        fpr = n_atk / total_w * 100.0
        print(f"Evaluasi (asumsi PCAP 100% Benign) -> Akurasi/Spesifisitas: {tnr:.2f}% ({n_ben}/{total_w}) | False Positive: {n_atk}/{total_w} ({fpr:.2f}%)")
    else:
        rec = n_atk / total_w * 100.0
        fnr = n_ben / total_w * 100.0
        print(f"Evaluasi (asumsi PCAP 100% Serangan) -> Recall: {rec:.2f}% ({n_atk}/{total_w}) | Lolos (FN): {n_ben}/{total_w} ({fnr:.2f}%)")
    if n_atk > 0:
        print("\nPasangan IP dengan window SERANGAN terbanyak:")
        print(df[df.prediksi == "SERANGAN"].groupby("pasangan_ip").size()
              .sort_values(ascending=False).head(5).to_string())
    print(f"\nHasil disimpan di: {a.out}")

if __name__ == "__main__":
    main()