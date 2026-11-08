import os
import glob
import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP, ICMP
from datetime import datetime, timezone, timedelta

# Zona waktu dataset
AST_TZ = timezone(timedelta(hours=-4))
ADT_TZ = timezone(timedelta(hours=-3))
BUANG_CELAH = True
ATTACKER_IPS = {"172.16.0.5"}
VICTIM_IPS = {"192.168.50.1", "192.168.50.4"}

def urutkan_angka_asli(filepath):
    fname = os.path.basename(filepath)
    bagian = fname.split("_")
    tanggal = bagian[0]
    nomor = int(bagian[-1]) if bagian[-1].isdigit() else 0
    return (tanggal, nomor)

def get_label(ts, src_ip, dst_ip, fname):
    # Traffic dari IP lain
    if not ((src_ip in ATTACKER_IPS and dst_ip in VICTIM_IPS) or
            (src_ip in VICTIM_IPS and dst_ip in ATTACKER_IPS)):
        return "Benign"

    # Pasangan attacker<->victim
    if "01-12-2018" in fname:
        dt = datetime.fromtimestamp(ts, tz=AST_TZ)
        hm = dt.hour * 60 + dt.minute
        if 10*60+35 <= hm <= 10*60+51: return "DrDoS_NTP"
        elif 10*60+52 <= hm <= 11*60+21: return "DrDoS_DNS"
        elif 11*60+22 <= hm <= 11*60+35: return "DrDoS_LDAP"
        elif 11*60+36 <= hm <= 11*60+49: return "DrDoS_MSSQL"
        elif 11*60+50 <= hm <= 12*60+11: return "DrDoS_NetBIOS"
        elif 12*60+12 <= hm <= 12*60+26: return "DrDoS_SNMP"
        elif 12*60+27 <= hm <= 12*60+44: return "DrDoS_SSDP"
        elif 12*60+45 <= hm <= 13*60+10: return "DrDoS_UDP"
        elif 13*60+11 <= hm <= 13*60+27: return "UDP-lag"
        elif 13*60+28 <= hm <= 13*60+34: return "Syn"
        elif hm >= 13*60+35: return "TFTP"
        else: return "CELAH"
    else:
        # Hari Kedua
        dt = datetime.fromtimestamp(ts, tz=ADT_TZ)
        hm = dt.hour * 60 + dt.minute
        if 9*60+43 <= hm <= 10*60+0: return "PortMap"
        elif 10*60+6 <= hm <= 10*60+31: return "NetBIOS"
        elif 10*60+36 <= hm <= 10*60+50: return "LDAP"
        elif 10*60+53 <= hm <= 11*60+24: return "MSSQL"
        elif 11*60+28 <= hm <= 11*60+59: return "UDP"
        elif 12*60+0 <= hm <= 12*60+28: return "UDP-lag"
        elif hm >= 12*60+29: return "Syn"
        else: return "CELAH"

def compute_44_features(pkts, label):
    n = len(pkts)
    if n == 0:
        return None
    ts_list = np.array([p['ts'] for p in pkts], dtype=np.float64)
    sizes = np.array([p['size'] for p in pkts], dtype=np.float64)
    hdrs = np.array([p['hdr_len'] for p in pkts], dtype=np.float64)
    ttls = np.array([p['ttl'] for p in pkts], dtype=np.float64)
    protos = np.array([p['proto'] for p in pkts], dtype=np.float64)
    out_sizes = np.array([p['size'] for p in pkts if p['dir'] == 1], dtype=np.float64)
    in_sizes = np.array([p['size'] for p in pkts if p['dir'] == 0], dtype=np.float64)
    flow_dur = max(float(ts_list[-1] - ts_list[0]), 1e-6)
    n_out, n_in = len(out_sizes), len(in_sizes)
    iats = np.diff(ts_list) if n > 1 else np.array([0.0])
    mean_out = float(np.mean(out_sizes)) if n_out > 0 else 0.0
    mean_in = float(np.mean(in_sizes)) if n_in > 0 else 0.0
    var_out = float(np.var(out_sizes)) if n_out > 0 else 0.0
    var_in = float(np.var(in_sizes)) if n_in > 0 else 0.0
    min_len = min(n_out, n_in)
    cov = float(np.cov(out_sizes[:min_len], in_sizes[:min_len])[0, 1]) if min_len > 1 else 0.0
    fin_c = sum(p['fin'] for p in pkts)
    syn_c = sum(p['syn'] for p in pkts)
    rst_c = sum(p['rst'] for p in pkts)
    psh_c = sum(p['psh'] for p in pkts)
    ack_c = sum(p['ack'] for p in pkts)
    ece_c = sum(p['ece'] for p in pkts)
    cwr_c = sum(p['cwr'] for p in pkts)

    return {
        "flow_duration": flow_dur,
        "Header_Length": float(np.mean(hdrs)),
        "Protocol Type": float(np.mean(protos)),
        "Duration": float(np.mean(ttls)),
        "Rate": n / flow_dur,
        "Srate": n_out / flow_dur,
        "Drate": n_in / flow_dur,
        "fin_flag_number": fin_c / n,
        "syn_flag_number": syn_c / n,
        "rst_flag_number": rst_c / n,
        "psh_flag_number": psh_c / n,
        "ack_flag_number": ack_c / n,
        "ece_flag_number": ece_c / n,
        "cwr_flag_number": cwr_c / n,
        "ack_count": float(ack_c),
        "syn_count": float(syn_c),
        "fin_count": float(fin_c),
        "HTTP": float(np.mean([p['http'] for p in pkts])),
        "HTTPS": float(np.mean([p['https'] for p in pkts])),
        "DNS": float(np.mean([p['dns'] for p in pkts])),
        "Telnet": float(np.mean([p['telnet'] for p in pkts])),
        "SMTP": float(np.mean([p['smtp'] for p in pkts])),
        "SSH": float(np.mean([p['ssh'] for p in pkts])),
        "IRC": float(np.mean([p['irc'] for p in pkts])),
        "TCP": float(np.mean([p['tcp'] for p in pkts])),
        "UDP": float(np.mean([p['udp'] for p in pkts])),
        "DHCP": float(np.mean([p['dhcp'] for p in pkts])),
        "ARP": 0.0,
        "ICMP": float(np.mean([p['icmp'] for p in pkts])),
        "IPv": 1.0,
        "LLC": 1.0,
        "Tot sum": float(np.sum(sizes)),
        "Min": float(np.min(sizes)),
        "Max": float(np.max(sizes)),
        "AVG": float(np.mean(sizes)),
        "Std": float(np.std(sizes)),
        "Tot size": float(np.mean(sizes)),
        "IAT": float(np.mean(iats)),
        "Number": float(n),
        "Magnitue": float(np.sqrt(max(mean_out + mean_in, 0.0))),
        "Radius": float(np.sqrt(max(var_out + var_in, 0.0))),
        "Covariance": cov,
        "Variance": float(var_in / var_out) if var_out > 0 else 0.0,
        "Weight": float(n_out * n_in),
        "Label": label
    }


def process_all_pcaps(folder="data_mentah", out_dir="csv_mentah", max_pkts_per_file=30000, window_size=50, batch_save_files=25):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "dataset_44fitur_raw.csv")
    out_parquet = os.path.join(out_dir, "dataset_44fitur_raw.parquet")
    # Bersihkan file CSV
    if os.path.exists(out_csv):
        os.remove(out_csv)

    # Ambil seluruh file dan urutkan secara natural
    all_files = sorted(glob.glob(os.path.join(folder, "SAT-*")), key=urutkan_angka_asli)
    total_files = len(all_files)
    print(f"Total PCAP terdeteksi: {total_files} Memproses seluruh file secara berurutan")
    print(f"BUANG_CELAH = {BUANG_CELAH}")
    buffer_rows = []
    total_baris_tersimpan = 0
    header_written = False

    # Counter diagnosa
    paket_benign_lain = 0
    paket_celah = 0
    for idx, fpath in enumerate(all_files, 1):
        fname = os.path.basename(fpath)
        buffers = {}
        pkt_count = 0
        try:
            with PcapReader(fpath) as pcap:
                for pkt in pcap:
                    if IP not in pkt:
                        continue
                    pkt_count += 1
                    ts = float(pkt.time)
                    src, dst = pkt[IP].src, pkt[IP].dst
                    proto = pkt[IP].proto
                    ttl = pkt[IP].ttl
                    size = len(pkt)
                    hdr_len = pkt[IP].ihl * 4 if hasattr(pkt[IP], 'ihl') and pkt[IP].ihl else 20
                    sport, dport = 0, 0
                    fin = syn = rst = psh = ack = ece = cwr = 0
                    is_tcp = is_udp = is_icmp = 0
                    if TCP in pkt:
                        is_tcp = 1
                        sport, dport = pkt[TCP].sport, pkt[TCP].dport
                        hdr_len += pkt[TCP].dataofs * 4 if pkt[TCP].dataofs else 20
                        fl = str(pkt[TCP].flags)
                        fin, syn, rst, psh, ack, ece, cwr = (
                            int('F' in fl), int('S' in fl), int('R' in fl),
                            int('P' in fl), int('A' in fl), int('E' in fl), int('C' in fl)
                        )
                    elif UDP in pkt:
                        is_udp = 1
                        sport, dport = pkt[UDP].sport, pkt[UDP].dport
                        hdr_len += 8
                    elif ICMP in pkt:
                        is_icmp = 1
                        hdr_len += 8
                    ports = {sport, dport}
                    label = get_label(ts, src, dst, fname)

                    if label == "CELAH":
                        paket_celah += 1
                        if BUANG_CELAH:
                            if pkt_count >= max_pkts_per_file:
                                break
                            continue
                        label = "Benign"
                    elif label == "Benign":
                        paket_benign_lain += 1
                    pair_key = (tuple(sorted([src, dst])), label)
                    direction = 1 if src <= dst else 0
                    if pair_key not in buffers:
                        buffers[pair_key] = []
                    buffers[pair_key].append({
                        'ts': ts, 'size': size, 'hdr_len': hdr_len, 'ttl': ttl, 'proto': proto,
                        'dir': direction, 'fin': fin, 'syn': syn, 'rst': rst, 'psh': psh,
                        'ack': ack, 'ece': ece, 'cwr': cwr, 'tcp': is_tcp, 'udp': is_udp, 'icmp': is_icmp,
                        'http': int(80 in ports or 8080 in ports),
                        'https': int(443 in ports),
                        'dns': int(53 in ports),
                        'telnet': int(23 in ports),
                        'smtp': int(25 in ports),
                        'ssh': int(22 in ports),
                        'irc': int(194 in ports or 6667 in ports),
                        'dhcp': int(67 in ports or 68 in ports)
                    })
                    if len(buffers[pair_key]) >= window_size:
                        row = compute_44_features(buffers[pair_key], label)
                        if row:
                            buffer_rows.append(row)
                        buffers[pair_key] = []
                    if pkt_count >= max_pkts_per_file:
                        break
            for (_, label), pkts_left in buffers.items():
                if len(pkts_left) >= 5:
                    row = compute_44_features(pkts_left, label)
                    if row:
                        buffer_rows.append(row)
            print(f"[{idx}/{total_files}] Selesai {fname} | Total di buffer: {len(buffer_rows)} | "
                  f"paket celah: {paket_celah:,} | paket benign murni: {paket_benign_lain:,}")
        except Exception as e:
            print(f"[{idx}/{total_files}] Skip {fname} karena error: {e}")

        # Simpan bertahap
        if idx % batch_save_files == 0 and buffer_rows:
            batch_df = pd.DataFrame(buffer_rows)
            batch_df.to_csv(out_csv, mode='a', index=False, header=not header_written)
            header_written = True
            total_baris_tersimpan += len(buffer_rows)
            print(f"{len(buffer_rows)} baris disimpan ke CSV (Total saat ini: {total_baris_tersimpan})")
            buffer_rows = []
    if buffer_rows:
        batch_df = pd.DataFrame(buffer_rows)
        batch_df.to_csv(out_csv, mode='a', index=False, header=not header_written)
        total_baris_tersimpan += len(buffer_rows)

    print(f"\nSemua file PCAP selesai diproses! Total akhir: {total_baris_tersimpan} baris.")
    print(f"Diagnosa label (jumlah paket): benign murni (bukan attacker<->victim) = {paket_benign_lain:,} | "
          f"attacker<->victim di luar jendela jam (celah) = {paket_celah:,} "
          f"({'DIBUANG' if BUANG_CELAH else 'dilabel Benign'})")
    # Konversi hasil akhir CSV ke Parquet
    print(f"Mengonversi {out_csv} ke {out_parquet}")
    final_df = pd.read_csv(out_csv)
    final_df.to_parquet(out_parquet, index=False)
    print("\nDistribusi Kelas Serangan Lengkap:")
    print(final_df['Label'].value_counts())
    print(f"\nSukses File utuh tersimpan di:\n1. {out_csv}\n2. {out_parquet}")


if __name__ == "__main__":
    process_all_pcaps(
        folder="data_mentah",
        out_dir="csv_mentah",
        max_pkts_per_file=30000,
        window_size=50,
        batch_save_files=25
    )