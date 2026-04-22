#!/usr/bin/env python3
"""
Sensör Client — FL istemcisi prototipi
TMC'ye JSON payload gönderir.
H3'te buraya model ağırlıkları eklenecek.
"""

import socket
import json
import sys

TMC_IP   = '10.0.0.10'
TMC_PORT = 9999

def send_to_tmc(sensor_id, round_num, anomaly_rate):
    msg = {
        'sensor_id'    : sensor_id,
        'round'        : round_num,
        'anomaly_rate' : anomaly_rate,
        # H3'te buraya 'weights': [...] eklenecek
    }
    
    payload = json.dumps(msg).encode()
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TMC_IP, TMC_PORT))
    s.sendall(payload)
    s.close()
    
    print(f"[Sensör {sensor_id}] TMC'ye {len(payload)} byte gönderildi.")

if __name__ == '__main__':
    # Komut satırından: python3 sensor_client.py <sensor_id> <anomaly_rate>
    sensor_id    = int(sys.argv[1])
    anomaly_rate = float(sys.argv[2])
    send_to_tmc(sensor_id, round_num=1, anomaly_rate=anomaly_rate)
