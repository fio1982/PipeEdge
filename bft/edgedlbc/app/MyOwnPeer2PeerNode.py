import copy
from os import read
from threading import TIMEOUT_MAX

import torch
from torch.utils.data.dataset import T
from p2pnetwork.node import Node
import operator
import time
import itertools
from utils import choice_model, test_inference, get_dataset, local_train, multi_krum, krum_distance, FedAvg, krum
import pickle
import subprocess
import ast
from global_sets import DATASET_NAME, MODEL_NAME, NODE_PATH, ATTACK_USERS, ATTACK_TYPE
from attack import add_gaussian_noise, sign_flipping_attack

TIMEOUT = 10
IDs = []
WORKER_SIZE = 0
# OFFSETS = {}
accepted_model = {}
record =  {}

class MyOwnPeer2PeerNode (Node):
    # Python class constructor
    def __init__(self, host, port, id=None, callback=None, max_connections=0, peer=0):
        super(MyOwnPeer2PeerNode, self).__init__(host, port, id, callback, max_connections, peer)
        print("MyPeer2PeerNode: Started")

    # all the methods below are called when things happen in the network.
    # implement your network node behavior to create the required functionality.

    def outbound_node_connected(self, node):
        print("outbound_node_connected (" + self.id + "): " + node.id)
        
    def inbound_node_connected(self, node):
        print("inbound_node_connected: (" + self.id + "): " + node.id)

    def inbound_node_disconnected(self, node):
        print("inbound_node_disconnected: (" + self.id + "): " + node.id)

    def outbound_node_disconnected(self, node):
        print("outbound_node_disconnected: (" + self.id + "): " + node.id)

    def node_message(self, node, data):
        # print("data: ", data)

        # get training data and start to train
        if isinstance(data, bytes):
            accuracy = 0
            received_data = pickle.loads(data)
            if received_data['type'] == 'train_data':
                print("%s receving train data from %s "% (self.id, node.id))
                print("+++ %s train data from %s transmission time: %s"% (self.id, node.id, time.time()*1000 - float(received_data['time'])))
                train_data = received_data

                reward = train_data['reward']
                deadline = train_data['deadline']
                
                picked_train_data = train_data['idx_train'][int(self.id)%len(train_data['idx_train'])]
                picked_teat_data = train_data['idx_test'][int(self.id)%len(train_data['idx_test'])]
                # print("picked_data: ", len(picked_train_data))

                # test initial model
                model = choice_model(MODEL_NAME)
                model.load_state_dict(choice_model(MODEL_NAME).state_dict())

                train_dataset, test_dataset = get_dataset(DATASET_NAME)

                acc = test_inference(model, test_dataset, picked_teat_data)
                accuracy = acc
                print("initial accuracy: ", accuracy)
                
                start = time.time()
                print("%s start training ...." % self.id)
                localModel = local_train(self, model, train_dataset, picked_train_data)
                
                if str(self.id) in ATTACK_USERS:
                    print("!!!!!!!!! attack: ", ATTACK_TYPE)
                    if ATTACK_TYPE == 'add_gaussian_noise':
                        start = time.time()
                        localModel = add_gaussian_noise(localModel, torch.device('cpu'), 100)
                    elif ATTACK_TYPE == 'sign_flipping_attack':
                        start = time.time()
                        localModel = sign_flipping_attack(localModel, p=-1)
                    elif ATTACK_TYPE == 'label_flipping_attack':
                        start = time.time()
                        localModel = local_train(self, model, train_dataset, picked_train_data, True)
                # start to train
                # start = time.time()
                # localModel = local_train(self, model, train_dataset, picked_train_data, is_flipping)
                print('%s train time %s ' % (self.id, time.time() - start))
                # print('+++ localModel type: ',  type(localModel))
                update = {
                    'type': 'update',
                    'model': localModel,
                    'worker_size': train_data['worker_size'],
                    'reward': reward,
                    'deadline': deadline,
                    'time': time.time()*1000
                }
                update_model = pickle.dumps(update)
                self.send_to_node(node, update_model)
                # self.node_message(node, update_model)

            elif received_data['type'] == 'update':
                print("%s receving updated model from %s "% (self.id, node.id))
                print("+++ update from %s transmission time: %s"% (node.id, time.time()*1000 - float(received_data['time'])))
                
                update = received_data
                # print('~~~~~~~~~~~~~~: received_data: ', len(received_data))
                update_model = update['model']
                start_time = update['time']
                reward = update['reward']
                deadline = update['deadline']

                localModel = choice_model(MODEL_NAME)
                localModel.load_state_dict(update_model)
                end_time = time.time()

                record[node.id] = str(start_time) + ',' + str(end_time) + ',' + reward + ',' + deadline

                workers_size = update['worker_size']
                print('worker_size: ', workers_size)
                
                accepted_model[int(node.id)] = copy.deepcopy(localModel)
                # print('accepted_model keys: ', accepted_model.keys())

                if len(accepted_model) == workers_size:
                    for i in range(self.peers):
                        if i+1 not in accepted_model:
                            accepted_model[i+1] = None
                    # print("*******accepted_model******** ", accepted_model.keys())
                    # comput krum score to give rewards
                    distances = krum_distance(accepted_model, self.peers)
                    _, scores = krum(workers_size,  workers_size//3, distances)
                    print('scores: ', scores)
                    
                    print('times: ', record)


                    localModel_set = multi_krum(accepted_model, workers_size, workers_size//3, self.peers)
                    print('localModel_set ', localModel_set)

                    self.addRecord(record, localModel_set, scores, node)
                            
                    localModels = [accepted_model[i].state_dict() for i in localModel_set]
                    globalModel = FedAvg(localModels)
                    
                    print('globalModel: ', len(globalModel))
                    accepted_model.clear()
                    record.clear()

                    model_test = choice_model(MODEL_NAME)
                    model_test.load_state_dict(globalModel)
                    _, test_dataset = get_dataset(DATASET_NAME)

                    acc = test_inference(model_test, test_dataset)
                    accuracy = acc
                    print("After accuracy: ", accuracy)

        # receive publish
        elif data["type"] == "msg_pub":
            print("server %s receive task: %s from  %s" % (self.id, data, node.id))
            res = {
                "type": "msg_res", 
                "id": self.id,
                "reward": data['rewards'],
                "deadline": data['deadline']
            }
            # give response
            self.send_to_node(node, res)
        # publisher recieve responses
        elif data["type"] == "msg_res" :
            print("server %s receive responses from: %s" % (self.id, node.id))
            IDs.append(node.id)
            # start to choose
            # sleep(TIMEOUT)
            # reputations = {}
            # with open('repu.txt') as f:
            #     for line in f:
            #         id, repu = line.replace('\n','').split(':')
            #         if id in IDs:
            #             reputations[id] = repu
                
            expected_res = 2*self.peers // 3
            expected_workers = self.peers // 2

            if sum(1 for ids in IDs) >= expected_res:
                received = IDs.copy()
                IDs.clear()

                # jsres = execute_js('../../client/mlClient.js')

                # if jsres.exitcode == 0:
                #     print('js execute successfully: ', jsres.stdout)
                # else:
                #     print('js execute unsuccessfully')
                reputations = self.getReputation(received)

                sorted_reputations = dict(sorted(reputations.items(), key=operator.itemgetter(1)))
                print("sorted_reputations: ", sorted_reputations)
                cut_reputations = dict(itertools.islice(sorted_reputations.items(), expected_workers))
                print("cut repu: ", cut_reputations)

                exclude_late = []
                exclude_lowRepu = []
                print("received: ", received)
                for i in range(1,10):
                    if not str(i) in received:
                        exclude_late.append(str(i))
                    if not str(i) in [k for k in cut_reputations.keys()]:
                        exclude_lowRepu.append(str(i))

                print("exclude_late: ", exclude_late)
                print("exclude_lowRepu: ", exclude_lowRepu)
                
                exclude = list(set(exclude_late).union(set(exclude_lowRepu)))
                
                print("exclude: ", exclude)
                
                WORKER_SIZE = self.peers - len(exclude)

                '''
                #get dataset
                train_dataset, test_dataset= get_dataset(DATASET_NAME)

                idx_train = {}
                idx_test = {}
                if DATASET_NAME == 'mnist':
                    idx_train = mnist_iid(train_dataset, WORKER_SIZE)
                    idx_test = mnist_iid(test_dataset, WORKER_SIZE)
                elif DATASET_NAME == 'cifar':
                    idx_train = cifar_iid(train_dataset, WORKER_SIZE)
                    idx_test = cifar_iid(test_dataset, WORKER_SIZE)
                
                # print("===== train_dataset len ====: ", len(train_dataset))
                # print("===== test_dataset len ====: ", len(test_dataset))
                # train_data = ["msg_train"]
                # self.send_to_nodes(train_data, exclude)

                train_data = {
                    'type': 'train_data',
                    'idx_train': idx_train,
                    'idx_test': idx_test
                }
                print("== sending data.......")
                print("sending train data: ", type(train_data))
                self.send_to_nodes(train_data, exclude)
                '''
                train_data = ['train_data', WORKER_SIZE, data['reward'], data['deadline']]
                self.send_to_nodes(train_data, exclude)
        # elif data["type"] == "train_data" :
        #     print("++++receive train_data+++++")
        #     train_data = self.recvall()
        #     print("received: ", len())
        # elif isinstance(data, list):
        #     train_data = pickle.loads(data)
        #     print("train_data: ", train_data[0])

    def getReputation(self, received):
        p = subprocess.Popen([NODE_PATH, '../../client/mlclient/getRepu.js', ','.join(received)], stdout=subprocess.PIPE)
        p.wait()
        out = p.stdout.read()
        reputations = ast.literal_eval(out.decode("utf-8"))
        # print('repu: ', repu)
        # print('repu size: ', len(repu))
        # reputations = {float(value) for value in repu.values()}
        for k, v in reputations.items():
            reputations[k] = float(v)
        print('reputations: ', reputations)
        return reputations

    def addRecord(self, record, localModel_set, scores, node):
        total = sum(scores.values())
        # print('total: ', total)
        transaction = ''
        for workerId in record:
            if int(workerId) in localModel_set:
                print("adding record.... ", workerId)
                tmp= record[workerId]
                tx = tmp.split(',')
                start_time = tx[0]
                end_time = tx[1]
                total_rewards = tx[2]
                deadline = tx[3]
                
                # print('scores[int(workerId)]/total: ', scores[int(workerId)]/total)
                # print('total_rewards: ', total_rewards)
                accept_reward = (scores[int(workerId)]/total)*float(total_rewards)
                # print(workerId)
                # print("start_time, end_time, reward, deadline", start_time, end_time, accept_reward, deadline)
                transaction += workerId+','+self.id+','+start_time+','+end_time+','+str(accept_reward)+','+deadline+','+str(scores[int(workerId)])+','               
        transaction = transaction.rstrip(',')
        p = subprocess.Popen([NODE_PATH, '../../client/mlclient/addRecord.js', transaction], stdout=subprocess.PIPE)
        # p.wait()
        # out = p.stdout.read()
        # print('transaction: ', out)

    def node_disconnect_with_outbound_node(self, node):
        print("node wants to disconnect with oher outbound node: (" + self.id + "): " + node.id)
        
    def node_request_to_stop(self):
        print("node is requested to stop (" + self.id + "): ")
        
