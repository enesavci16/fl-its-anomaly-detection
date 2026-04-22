#!/usr/bin/env python3
"""
ITS Federated Learning Topolojisi
4 sensör düğümü (FL istemcisi) + 1 TMC (FL sunucusu) + 1 OpenFlow switch
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel

class ITSTopo(Topo):
    """
    Star topoloji:
    h1, h2, h3, h4 (sensörler) ve tmc (TMC) → hepsi s1'e bağlı
    """
    def build(self):
        # Switch ekle
        s1 = self.addSwitch('s1')

        # TMC host'u ekle
        tmc = self.addHost('tmc', ip='10.0.0.10/24')

        # 4 sensör host'u ekle
        hosts = []
        for i in range(1, 5):
            h = self.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
            hosts.append(h)

        # Tüm host'ları switch'e bağla
        self.addLink(tmc, s1)
        for h in hosts:
            self.addLink(h, s1)


def run():
    setLogLevel('info')

    topo = ITSTopo()

    # RemoteController: os-ken'i ayrı terminalde çalıştıracağız
    net = Mininet(
        topo=topo,
        controller=RemoteController('c0', ip='127.0.0.1', port=6653)
    )

    net.start()

    print("\n=== ITS Topolojisi Başlatıldı ===")
    print("Düğümler: h1, h2, h3, h4 (sensörler) + tmc (TMC sunucusu)")
    print("Switch: s1 (OpenFlow 1.3, os-ken tarafından yönetiliyor)")
    print("Çıkmak için: exit\n")

    CLI(net)      # Mininet komut satırı — pingall burada çalıştırılır
    net.stop()


if __name__ == '__main__':
    run()

