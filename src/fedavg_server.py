#!/usr/bin/env python3
"""
FedAvg Server — TMC (Trafik Yönetim Merkezi)
1. Her client'tan yerel modeli alır
2. FedAvg aggregation uygular
3. Global modeli geri gönderir
"""

import socket
import json
import threading
import struct
from fedavg_utils import bytes_to_model, model_to_bytes, fedavg_aggregate

HOST = '0.0.0.0'
PORT = 9999
N_CLIENTS = 4
N_ROUNDS = 3

def recv_msg(conn):
    """Önce 4 byte uzunluk başlığı al, sonra mesajı al"""
    raw_len = conn.recv(4)
    if not raw_len:
        return None
    msg_len = struct.unpack('>I', raw_len)[0]
    data = b""
    while len(data) < msg_len:
        chunk = conn.recv(min(4096, msg_len - len(data)))
        if not chunk:
            break
        data += chunk
    return json.loads(data.decode())

def send_msg(conn, msg):
    """Önce 4 byte uzunluk başlığı gönder, sonra mesajı gönder"""
    data = json.dumps(msg).encode()
    conn.sendall(struct.pack('>I', len(data)))
    conn.sendall(data)

def run_round(round_num):
    """Tek bir FL turu"""
    print(f"\n{'='*40}")
    print(f"[TMC] Round {round_num} başladı")
    print(f"{'='*40}")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(N_CLIENTS)
    
    client_data = []
    conns = []
    
    # 4 client'ı bekle
    for i in range(N_CLIENTS):
        conn, addr = server.accept()
        conns.append(conn)
        msg = recv_msg(conn)
        print(f"[TMC] Sensör {msg['sensor_id']} bağlandı "
              f"| {msg['n_samples']} örnek "
              f"| payload: {len(json.dumps(msg).encode())} byte")
        client_data.append(msg)
    
    # Modelleri deserialize et
    models = [bytes_to_model(d['model']) for d in client_data]
    n_samples = [d['n_samples'] for d in client_data]
    
    # FedAvg aggregation
    print(f"\n[TMC] FedAvg aggregation yapılıyor...")
    global_model = fedavg_aggregate(models, n_samples)
    print(f"[TMC] Global offset hazır — "
          f"offset_={global_model:.6f}")
    
    # Global modeli tüm client'lara gönder
    # Global offset'i tüm client'lara gönder
    total_bytes_sent = 0
    for conn in conns:
        msg = {'global_offset': global_model}  # global_model artık float
        send_msg(conn, msg)
        total_bytes_sent += len(json.dumps(msg).encode())
        conn.close()
    
    server.close()
    
    print(f"[TMC] Round {round_num} tamamlandı")
    print(f"[TMC] Toplam iletişim maliyeti: "
          f"{total_bytes_sent/1024:.2f} KB (server→client)")
    
    return global_model

def main():
    print(f"[TMC] FedAvg Server başlatıldı — {N_ROUNDS} round")
    for r in range(1, N_ROUNDS + 1):
        global_model = run_round(r)
    print(f"\n[TMC] Tüm roundlar tamamlandı. ✓")

if __name__ == '__main__':
    main()
