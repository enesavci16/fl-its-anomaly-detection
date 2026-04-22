# FL + ITS Proje — İlerleme Günlüğü

## Mevcut Durum Özeti
**Aktif Hafta:** H4 (S2+S3 tamamlandı, takvimin önünde)
**ASYU 2026 Deadline:** 1 Haziran 2026 (~5 hafta)
**Son Güncelleme:** 22 Nisan 2026
**Bu Hafta:** H2 + H3 tamamlandı ✓

---

## Hafta 2 — Mininet Topoloji (21-27 Nisan 2026)
### Durum: TAMAMLANDI ✓
- [x] its_topo.py: 4 sensör + TMC + s1 switch
- [x] pingall: 0% dropped (20/20) ✓
- [x] TCP socket demo çalıştı

## Hafta 3 — FedAvg (28 Nisan-4 Mayıs 2026)
### Durum: TAMAMLANDI ✓ (erken — 22 Nisan)
- [x] fedavg_utils.py, fedavg_server.py, fedavg_client.py
- [x] S2 IID FL tamamlandı
- [x] S3 Non-IID FL tamamlandı
- [x] GitHub push edildi

---

## Ablasyon Sonuçları

| Senaryo | Ort. F1 | S1 Farkı |
|---------|---------|----------|
| S1 Merkezi | 0.7715 | — |
| S2 IID FL | 0.7028 | -0.0687 |
| S3 Non-IID | 0.5843 | -0.1872 |

### Kritik Bulgular
1. S2 vs S1: %8.9 F1 kaybı — kabul edilebilir trade-off
2. S3 vs S2: %11.8 düşüş — client drift kanıtlandı
3. Server→client maliyet: 0.15 KB (federated threshold calibration)
4. Sensör 240: global F1 < yerel F1 — Non-IID etkisi bireysel kanıt
