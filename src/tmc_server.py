#!/usr/bin/env python3
"""
TMC Server — FL aggregation sunucusu prototipi
Her sensör düğümünden JSON mesaj alır, ekrana basar.
H3'te buraya FedAvg aggregation gelecek.
"""

import socket
import json
import threading

HOST = '0.0.0.0'   # Her interface'den gelen bağlantıyı kabul et
PORT = 9999

def handle_client(conn, addr):
    """Her client için ayrı thread — aynı anda 4 sensör bağlanabilsin"""
    print(f"[TMC] Bağlantı geldi: {addr}")
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        
        msg = json.loads(data.decode())
        print(f"[TMC] Sensör {msg['sensor_id']} mesajı:")
        print(f"      Round: {msg['round']}")
        print(f"      Anomali oranı: {msg['anomaly_rate']:.4f}")
        print(f"      Payload boyutu: {len(data)} byte")
    
    except Exception as e:
        print(f"[TMC] Hata: {e}")
    finally:
        conn.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"[TMC] {PORT} portunda dinleniyor...")
    
    while True:
        conn, addr = server.accept()
        t = threading.Thread(target=handle_client, args=(conn, addr))
        t.start()

if __name__ == '__main__':
    main()
