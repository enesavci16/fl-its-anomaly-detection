#!/usr/bin/env python3
"""
FedAvg Client — Sensör düğümü (FL istemcisi)
1. Yerel veriyle Isolation Forest eğitir
2. Modeli TMC'ye gönderir
3. Global modeli alır, F1 skorunu hesaplar
"""

import socket
import json
import struct
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

# fedavg_utils'i import etmek için src dizinini path'e ekle
sys.path.append('/home/enes/fl-its/src')
from fedavg_utils import model_to_bytes, bytes_to_model

TMC_IP   = '127.0.0.1'
TMC_PORT = 9999

# Sensör ID → veri yolu eşleştirmesi
SENSOR_MAP = {
    278: '/home/enes/fl-its-data/data/splits/iid/client_0.csv',
    240: '/home/enes/fl-its-data/data/splits/iid/client_1.csv',
    71:  '/home/enes/fl-its-data/data/splits/iid/client_2.csv',
    298: '/home/enes/fl-its-data/data/splits/iid/client_3.csv',
}

def recv_msg(conn):
    raw_len = conn.recv(4)
    msg_len = struct.unpack('>I', raw_len)[0]
    data = b""
    while len(data) < msg_len:
        chunk = conn.recv(min(4096, msg_len - len(data)))
        if not chunk:
            break
        data += chunk
    return json.loads(data.decode())

def send_msg(conn, msg):
    data = json.dumps(msg).encode()
    conn.sendall(struct.pack('>I', len(data)))
    conn.sendall(data)

def load_data(sensor_id):
    """Sensörün yerel verisini yükle"""
    df = pd.read_csv(SENSOR_MAP[sensor_id])
    features = ['flow', 'speed', 'occupancy']
    
    train = df[df['split'] == 'train']
    test  = df[df['split'] == 'test']
    
    X_train = train[features].values
    X_test  = test[features].values
    y_test  = test['is_anomaly'].values
    
    return X_train, X_test, y_test

def train_local_model(X_train, contamination=0.00826):
    """Yerel Isolation Forest eğit — S1 baseline ile aynı contamination"""
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    model.fit(X_train)
    return model

def evaluate(model, X_test, y_test):
    """F1 skoru hesapla"""
    # Isolation Forest: -1 anomali, 1 normal → 0/1'e çevir
    preds = model.predict(X_test)
    preds_binary = (preds == -1).astype(int)
    return f1_score(y_test, preds_binary, zero_division=0)

def run_client(sensor_id, round_num):
    """Tek bir FL turu — client tarafı"""
    print(f"\n[Sensör {sensor_id}] Round {round_num} başladı")
    
    # Veri yükle
    X_train, X_test, y_test = load_data(sensor_id)
    print(f"[Sensör {sensor_id}] Veri yüklendi: "
          f"{len(X_train)} train, {len(X_test)} test")
    
    # Yerel model eğit
    local_model = train_local_model(X_train)
    local_f1 = evaluate(local_model, X_test, y_test)
    print(f"[Sensör {sensor_id}] Yerel F1: {local_f1:.4f}")
    
    # TMC'ye bağlan ve modeli gönder
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TMC_IP, TMC_PORT))
    
    model_bytes = model_to_bytes(local_model)
    msg = {
        'sensor_id': sensor_id,
        'round':     round_num,
        'n_samples': len(X_train),
        'local_f1':  local_f1,
        'model':     model_bytes
    }
    send_msg(s, msg)
    print(f"[Sensör {sensor_id}] Model gönderildi: "
          f"{len(model_bytes)/1024:.2f} KB")
    
    # Global modeli al
    # Global offset'i al
    response = recv_msg(s)
    s.close()
    
    # Yerel model tree'leri koru, sadece eşiği güncelle
    local_model.offset_ = response['global_offset']
    global_f1 = evaluate(local_model, X_test, y_test)
    print(f"[Sensör {sensor_id}] Global model F1: {global_f1:.4f}")
    
    return local_f1, global_f1

def main():
    sensor_id = int(sys.argv[1])
    N_ROUNDS  = 3
    
    results = []
    for r in range(1, N_ROUNDS + 1):
        local_f1, global_f1 = run_client(sensor_id, r)
        results.append({
            'round': r,
            'local_f1': local_f1,
            'global_f1': global_f1
        })
    
    print(f"\n[Sensör {sensor_id}] Tüm roundlar tamamlandı:")
    for r in results:
        print(f"  Round {r['round']}: "
              f"yerel={r['local_f1']:.4f}, "
              f"global={r['global_f1']:.4f}")

if __name__ == '__main__':
    main()
