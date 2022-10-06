# Python implementation of a peer-to-peer decentralized network


# Evolution of the software


# Design


## You have two options


## Option 1: Implement your p2p application by extending Node and NodeConnection classes


## Events that can occur

### outbound_node_connected
The node connects with another node - ````node.connect_with_node('127.0.0.1', 8002)```` - and the connection is successful. While the basic functionality is to exchange the node id's, no user data is involved.

### inbound_node_connected
Another node has made a connection with this node and the connection is successful. While the basic functionality is to exchange the node id's, no user data is involved.

### outbound_node_disconnected
A node, to which we had made a connection in the past, is disconnected.

### inbound_node_disconnected
A node, that had made a connection with us in the past, is disconnected.

### node_message
A node - ```` connected_node ```` - sends a message. At this moment the basic functionality expects JSON format. It tries to decode JSON when the message is received. If it is not possible, the message is rejected.

### node_disconnect_with_outbound_node
The application actively wants to disconnect the outbound node, a node with which we had made a connection in the past. You could send some last message to the node, that you are planning to disconnect, for example.

### node_request_to_stop
The main node, also the application, is stopping itself. Note that the variable connected_node is empty, while there is no connected node involved.

# Debugging

When things go wrong, you could enable debug messages of the Node class. The class shows these messages in the console and shows all the details of what happens within the class. To enable debugging for a node, use the code example below.

````python
node = Node("127.0.0.1", 10001)
node.debug = True
````

# Unit testing

Several unit tests have been implemented to make sure all the functionality of the provided classes are working correctly. To run these tests, you can use the following code:

````bash
$ python setup.py test
````

# Examples

Examples are available in the github repository of this project: https://github.com/macsnoeren/python-p2p-network. All examples can be found in the directory examples.

# Node and NodeConnection class                                       

See the Python documentation for all the details of these classes.

# Show case: SecureNode
As a show case, I have created the SecureNode class that extends the Node class. This node uses JSON, hashing and signing to communicate between the nodes. My main thought with this secure node is to be able to exchange data securely with each other and give others permissions to read the data, for example. You are the owner of your data! Anyway, some project that I am currently working on. See the documentation of this specific class file.

````python
import sys
import time

from p2pnetwork.securenode import SecureNode

node = SecureNode("127.0.0.1", 10001)
time.sleep(1)

node.start()
````
An example node that uses SecureNode class is found in the example directory on github: ````secure_node.py````.


torch 1.7.1
torchvision 0.8.2
torchaudio 0.7.1
torchtext 0.8.1