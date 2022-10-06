import sys
import time
sys.path.insert(0, '..') # Import the files where the modules are located

from MyOwnPeer2PeerNode import MyOwnPeer2PeerNode
from global_sets import WORKERS


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

nodes = [node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8, node_9, node_10]
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

time.sleep(1)
def connections(node):
    for port in ports:
        if int(node.id) == int(str(port)[-2:]):
            break
        node.connect_with_node('127.0.0.1', port)
        
for node in nodes:
    connections(node)       

# node_1.connect_with_node('127.0.0.1', 8002)
# node_2.connect_with_node('127.0.0.1', 8003)
# node_3.connect_with_node('127.0.0.1', 8001)

# node_4.connect_with_node('127.0.0.1', 8001)
# node_5.connect_with_node('127.0.0.1', 8003)
# node_6.connect_with_node('127.0.0.1', 8005)

# node_7.connect_with_node('127.0.0.1', 8004)
# node_8.connect_with_node('127.0.0.1', 8005)
# node_9.connect_with_node('127.0.0.1', 8006)

# node_10.connect_with_node('127.0.0.1', 8009)

time.sleep(2)

# publish a job
node_1.send_to_nodes({"type":"msg_pub", "rewards":"200", "deadline":"10000", "workers":WORKERS, "time": time.time()*1000})


# exclude = [node_2.id]

# node_1.send_to_nodes(b'')

# time.sleep(100)

# node_1.stop()
# node_2.stop()
# node_3.stop()
# node_4.stop()
# node_5.stop()
# node_6.stop()
# node_7.stop()
# node_8.stop()
# node_9.stop()
# node_10.stop()

# print('end test')
