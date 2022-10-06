import sys
import time
sys.path.insert(0, '..') # Import the files where the modules are located

from MyOwnPeer2PeerNode import MyOwnPeer2PeerNode
ports = [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010]

node_1 = MyOwnPeer2PeerNode("127.0.0.1", 8001, 1, peer=10)
node_2 = MyOwnPeer2PeerNode("127.0.0.1", 8002, 2, peer=10)
node_3 = MyOwnPeer2PeerNode("127.0.0.1", 8003, 3, peer=10)
node_4 = MyOwnPeer2PeerNode("127.0.0.1", 8004, 4, peer=10)
node_5 = MyOwnPeer2PeerNode("127.0.0.1", 8005, 5, peer=10)
node_6 = MyOwnPeer2PeerNode("127.0.0.1", 8006, 6, peer=10)
node_7 = MyOwnPeer2PeerNode("127.0.0.1", 8007, 7, peer=10)
node_8 = MyOwnPeer2PeerNode("127.0.0.1", 8008, 8, peer=10)
node_9 = MyOwnPeer2PeerNode("127.0.0.1", 8009, 9, peer=10)
node_10 = MyOwnPeer2PeerNode("127.0.0.1", 8010, 10, peer=10)
node_11 = MyOwnPeer2PeerNode("127.0.0.1", 8011, 11, peer=11)
node_12 = MyOwnPeer2PeerNode("127.0.0.1", 8012, 12, peer=12)
node_13 = MyOwnPeer2PeerNode("127.0.0.1", 8013, 13, peer=13)
node_14 = MyOwnPeer2PeerNode("127.0.0.1", 8014, 14, peer=14)
node_15 = MyOwnPeer2PeerNode("127.0.0.1", 8015, 15, peer=15)
node_16 = MyOwnPeer2PeerNode("127.0.0.1", 8016, 16, peer=16)
node_17 = MyOwnPeer2PeerNode("127.0.0.1", 8017, 17, peer=17)
node_18 = MyOwnPeer2PeerNode("127.0.0.1", 8018, 18, peer=18)
node_19 = MyOwnPeer2PeerNode("127.0.0.1", 8019, 19, peer=19)
node_20 = MyOwnPeer2PeerNode("127.0.0.1", 8020, 20, peer=20)
node_21 = MyOwnPeer2PeerNode("127.0.0.1", 8021, 16, peer=21)
node_22 = MyOwnPeer2PeerNode("127.0.0.1", 8022, 17, peer=22)
node_23 = MyOwnPeer2PeerNode("127.0.0.1", 8023, 18, peer=23)
node_24 = MyOwnPeer2PeerNode("127.0.0.1", 8024, 19, peer=24)
node_25 = MyOwnPeer2PeerNode("127.0.0.1", 8025, 20, peer=25)
node_26 = MyOwnPeer2PeerNode("127.0.0.1", 8026, 16, peer=26)
node_27 = MyOwnPeer2PeerNode("127.0.0.1", 8027, 17, peer=27)
node_28 = MyOwnPeer2PeerNode("127.0.0.1", 8028, 18, peer=28)
node_29 = MyOwnPeer2PeerNode("127.0.0.1", 8029, 19, peer=29)
node_30 = MyOwnPeer2PeerNode("127.0.0.1", 8030, 20, peer=30)

nodes = [node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8, node_9, node_10, node_11, node_12, node_13, node_14, node_15, node_16, node_17, node_18, node_19, node_20, node_21, node_22, node_23, node_24, node_25, node_26, node_27, node_28, node_29, node_30]
time.sleep(1)

node_1.start()
node_2.start()
node_3.start()
node_4.start()
node_5.start()
node_6.start()
node_7.start()
node_8.start()
node_9.start()
node_10.start()
node_11.start()
node_12.start()
node_13.start()
node_14.start()
node_15.start()
node_16.start()
node_17.start()
node_18.start()
node_19.start()
node_20.start()
node_21.start()
node_22.start()
node_23.start()
node_24.start()
node_25.start()
node_26.start()
node_27.start()
node_28.start()
node_29.start()
node_30.start()

time.sleep(20)
def connections(node):
    for port in ports:
        if int(node.id) == int(str(port)[-2:]):
            break
        node.connect_with_node('127.0.0.1', port)
        
for node in nodes:
    connections(node)       


time.sleep(20)

# publish a job
node_2.send_to_nodes({"type":"msg_pub", "rewards":"200", "deadline":"10000"})

