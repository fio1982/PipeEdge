import socket
import time
import threading
import json
import os
import pickle
import sys
# sys.path.append("..")
from app.utils import get_dataset, mnist_iid, cifar_iid
from global_sets import DATASET_NAME

"""
Author : Maurice Snoeren <macsnoeren(at)gmail.com>
Version: 0.3 beta (use at your own risk)
Date: 7-5-2020

Python package p2pnet for implementing decentralized peer-to-peer network applications
"""
class NodeConnection(threading.Thread):
    """The class NodeConnection is used by the class Node and represent the TCP/IP socket connection with another node. 
       Both inbound (nodes that connect with the server) and outbound (nodes that are connected to) are represented by
       this class. The class contains the client socket and hold the id information of the connecting node. Communication
       is done by this class. When a connecting node sends a message, the message is relayed to the main node (that created
       this NodeConnection in the first place).
       
       Instantiates a new NodeConnection. Do not forget to start the thread. All TCP/IP communication is handled by this 
       connection.
        main_node: The Node class that received a connection.
        sock: The socket that is assiociated with the client connection.
        id: The id of the connected node (at the other side of the TCP/IP connection).
        host: The host/ip of the main node.
        port: The port of the server of the main node."""

    def __init__(self, main_node, sock, id, host, port):
        """Instantiates a new NodeConnection. Do not forget to start the thread. All TCP/IP communication is handled by this connection.
            main_node: The Node class that received a connection.
            sock: The socket that is assiociated with the client connection.
            id: The id of the connected node (at the other side of the TCP/IP connection).
            host: The host/ip of the main node.
            port: The port of the server of the main node."""

        super(NodeConnection, self).__init__()

        self.host = host
        self.port = port
        self.main_node = main_node
        self.sock = sock
        self.terminate_flag = threading.Event()

        # The id of the connected node
        self.id = str(id) # Make sure the ID is a string

        # End of transmission character for the network streaming messages.
        self.EOT_CHAR = 0x04.to_bytes(1, 'big')

        # Datastore to store additional information concerning the node.
        self.info = {}

        # Use socket timeout to determine problems with the connection
        self.sock.settimeout(500.0)

        self.main_node.debug_print("NodeConnection.send: Started with client (" + self.id + ") '" + self.host + ":" + str(self.port) + "'")

    def send(self, data, encoding_type='utf-8'):
        """Send the data to the connected node. The data can be pure text (str), dict object (send as json) and bytes object.
           When sending bytes object, it will be using standard socket communication. A end of transmission character 0x04 
           utf-8/ascii will be used to decode the packets ate the other node. When the socket is corrupted the node connection
           is closed."""
        if isinstance(data, str):
            try:
                self.sock.sendall( data.encode(encoding_type) + self.EOT_CHAR )

            except Exception as e: # Fixed issue #19: When sending is corrupted, close the connection
                self.main_node.debug_print("nodeconnection send: Error sending data to node: " + str(e))
                self.stop() # Stopping node due to failure

        elif isinstance(data, dict):
            # if data['type'] == 'train_data':
            #     data_pickle = pickle.dumps(data)
            #     # print("++++train_data data_pickle type: ", type(data_pickle))
            #     self.sock.sendall(data_pickle)
            # else: 
            try:
                json_data = json.dumps(data)
                json_data = json_data.encode(encoding_type) + self.EOT_CHAR
                self.sock.sendall(json_data)

            except TypeError as type_error:
                self.main_node.debug_print('This dict is invalid')
                self.main_node.debug_print(type_error)

            except Exception as e: # Fixed issue #19: When sending is corrupted, close the connection
                self.main_node.debug_print("nodeconnection send: Error sending data to node: " + str(e))
                self.stop() # Stopping node due to failure
        # elif isinstance(data, list):
        #     print("tuple data")
        #     tmp = pickle.dumps(data)
        #     self.sock.sendall(tmp)
        elif isinstance(data, list) :
            # filename = "t10k-images-idx3-ubyte"
            # # offsets = data[1]
            # total = self.readFile(filename)
            # # print("total read: ", len(total))
            # # split data send to servers accordingly
            # self.sock.sendall(total)
            #get dataset
            WORKER_SIZE = data[1]
            reward = data[2]
            deadline = data[3]

            # DATASET_NAME = 'mnist'
            train_dataset, test_dataset= get_dataset(DATASET_NAME)
            idx_train = {}
            idx_test = {}
            if DATASET_NAME == 'mnist':
                idx_train = mnist_iid(train_dataset, WORKER_SIZE*10)
                idx_test = mnist_iid(test_dataset, WORKER_SIZE*10)
            elif DATASET_NAME == 'cifar':
                idx_train = cifar_iid(train_dataset, WORKER_SIZE*10)
                idx_test = cifar_iid(test_dataset, WORKER_SIZE*10)
            elif DATASET_NAME == 'cifar100':
                idx_train = cifar_iid(train_dataset, WORKER_SIZE*10)
                idx_test = cifar_iid(test_dataset, WORKER_SIZE*10)
        
            train_data = {
                'type': 'train_data',
                'idx_train': idx_train,
                'idx_test': idx_test,
                'worker_size': WORKER_SIZE,
                'reward': reward,
                'deadline': deadline,
                'time': time.time()*1000   
            }
            # with_time = {
            #     'type': 'train_data',
            #     'data': train_data,
            #     'time': time.time()*1000
            # }
            print("sending data....... ")
            # pickled = pickle.dumps(with_time)
            pickled = pickle.dumps(train_data)
            self.sock.sendall(pickled)

        else:
            print('sending updates: ', len(data))
            self.sock.sendall(data)
            # self.main_node.debug_print('datatype used is not valid plese use str, dict (will be send as json) or bytes')
    
    def readFile(self, filename):
        total = bytearray(b'')
        with open(filename, "rb") as file:
            for line in file:
                # print("l : ", type(line))
                total += line
        return total

    def recvall(self):
        BUFF_SIZE = 4096 # 4 KiB
        data = b''
        while True:
            part = self.sock.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                break # either 0 or end of data

        return data

    # This method should be implemented by yourself! We do not know when the message is
    # correct.
    # def check_message(self, data):
    #         return True

    # Stop the node client. Please make sure you join the thread.
    def stop(self):
        """Terminates the connection and the thread is stopped."""
        self.terminate_flag.set()

    def parse_packet(self, packet):
        """Parse the packet and determines wheter it has been send in str, json or byte format. It returns
           the according data."""
        try:
            packet_decoded = packet.decode('utf-8')

            try:
                return json.loads(packet_decoded)

            except json.decoder.JSONDecodeError:
                return packet_decoded

        except UnicodeDecodeError:
            return packet

    # Required to implement the Thread. This is the main loop of the node client.
    def run(self):
        """The main loop of the thread to handle the connection with the node. Within the
           main loop the thread waits to receive data from the node. If data is received 
           the method node_message will be invoked of the main node to be processed."""          
        buffer = b'' # Hold the stream that comes in!

        while not self.terminate_flag.is_set():
            chunk = b''
            # tmp = []
            try:
                # chunk = self.sock.recv(4096*10000*5) 
                # BUFF_SIZE = 4096 # 4 KiB
                # while True:
                #     part = self.sock.recv(BUFF_SIZE) 
                #     chunk += part
                #     if len(part) < BUFF_SIZE:
                #         break # either 0 or end of data

                chunk = self.recvall()

                # BUFF_SIZE = 4096 # 4 KiB
                # while True:
                #     part = self.sock.recv(BUFF_SIZE) 
                #     if not part: break
                #     tmp.append(part)
                # chunk = b"".join(tmp)

            except socket.timeout:
                self.main_node.debug_print("NodeConnection: timeout")

            except Exception as e:
                self.terminate_flag.set() # Exception occurred terminating the connection
                self.main_node.debug_print('Unexpected error')
                self.main_node.debug_print(e)

            # BUG: possible buffer overflow when no EOT_CHAR is found => Fix by max buffer count or so?
            # print("!!!!!!!!!!!!chunk: ", len(chunk))
            # print("!!!!!!!!!!!!EOT_CHAR: ", len(self.EOT_CHAR))
            if len(chunk) >= 4096:
                self.main_node.node_message(self, chunk)
                # temp = pickle.loads(chunk)
                # if temp['type'] == 'train_data':
                #     pickled = pickle.dumps(temp)
                #     # self.sock.sendall(pickled)
                #     self.main_node.node_message(self, pickled)
                # else:
                #     with_time = {
                #         'type': 'update',
                #         'data': temp
                #     }
                #     pickled = pickle.dumps(with_time)
                #     self.main_node.node_message(self, pickled)


            elif chunk != b'':
                buffer += chunk
                eot_pos = buffer.find(self.EOT_CHAR)
                # print("!!!!!!!!!!!!eot_pos: ", eot_pos)
                while eot_pos > 0:
                    packet = buffer[:eot_pos]
                    buffer = buffer[eot_pos + 1:]
                    # print("!!!!!!!!!!!!buffer: ", len(buffer))
                    self.main_node.message_count_recv += 1
                    self.main_node.node_message( self, self.parse_packet(packet) )

                    eot_pos = buffer.find(self.EOT_CHAR)
                    # print("!!!!!!!!!!!!eot_pos after: ", eot_pos)
                    break

            time.sleep(0.01)

        # IDEA: Invoke (event) a method in main_node so the user is able to send a bye message to the node before it is closed?
        self.sock.settimeout(None)
        self.sock.close()
        self.main_node.node_disconnected( self ) # Fixed issue #19: Send to main_node when a node is disconnected. We do not know whether it is inbounc or outbound.
        self.main_node.debug_print("NodeConnection: Stopped")

    def set_info(self, key, value):
        self.info[key] = value

    def get_info(self, key):
        return self.info[key]

    def __str__(self):
        return 'NodeConnection: {}:{} <-> {}:{} ({})'.format(self.main_node.host, self.main_node.port, self.host, self.port, self.id)

    def __repr__(self):
        return '<NodeConnection: Node {}:{} <-> Connection {}:{}>'.format(self.main_node.host, self.main_node.port, self.host, self.port)
