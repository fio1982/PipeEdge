import copy
import imp
import os
from app.models import modelC

# import torch
# from torch.utils.data.dataset import T
from p2pnetwork.node import Node
import operator
import time
import itertools
from utils import choose_model_partition, test_inference, get_dataset, local_train, multi_krum, krum_distance, FedAvg, krum
import pickle
import subprocess
import ast
from global_sets import CANDIDATES, DATASET_NAME, NODE_PATH, WORKERS, MODEL_PARTITION_NAME

from mp.awd_lm.utils import loadData, buildModel, localtrain
from mp.awd_lm import awdmodelpartition

from mp.transformer.train import train, getTransformerModel, optimizer, criterion
from mp.transformer.conf import epoch, inf, clip
from mp.transformer.data import train_iter

TIMEOUT = 10
IDs = []
delays = {}
accepted_model = {}
record =  {}
awd_model = None

'''
publisher send worker1 with all workers' id
duplicate: multi-workers train the same partition
'''

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

    # def recvBigData(self):
    #     print('++++++ recv big data ++++++++')
    #     BUFF_SIZE = 4096 # 4 KiB
    #     buf = b''
    #     while True:
    #         part = self.sock.recv(BUFF_SIZE)
    #         buf += part
    #         if len(part) < BUFF_SIZE:
    #             break

    #     return buf
    
    def node_message(self, node, data):
        # print("data: ", data)
        # get training data and start to train
        
        if isinstance(data, bytes):
            global awd_model
            accuracy = 0
            if len(data) > 4096:
                time.sleep(60)
            print('++++++ data received %s++++++++' % len(data))
            received_data = pickle.loads(data)
            if received_data['type'] == 'train_data':
                print("%s receved train data from %s to train %s"% (self.id, node.id, MODEL_PARTITION_NAME))
                print("+++ %s train data from %s transmission time: %s"% (self.id, node.id, time.time()*1000 - float(received_data['time'])))
                train_data = received_data
                reward = train_data['reward']
                deadline = train_data['deadline']
                localModel = None
                if MODEL_PARTITION_NAME == 'awd':
                    print("========= start on %s ==========="% MODEL_PARTITION_NAME)
                    path = __file__
                    path = os.path.dirname(path)
                    current_path = os.path.join(path, 'mp','awd_lm','data','penn')
                    train_data, corpus, val_data, test_data = loadData(current_path)
                    localModel, awd_model = localtrain(awdmodelpartition, train_data, corpus, self.id)
                elif MODEL_PARTITION_NAME == 'transformer':
                    print("========= start on %s ==========="% MODEL_PARTITION_NAME)
                    # run(total_epoch=epoch, best_loss=inf)
                    start = time.time()*1000
                    localModel = train(getTransformerModel(), train_iter, optimizer, criterion, clip, self.id)
                    print('%s train time %s ' % (self.id, time.time()*1000 - start))
                else:
                    print("========= start on %s ==========="% MODEL_PARTITION_NAME)
                    
                    picked_train_data = train_data['idx_train'][int(self.id)%len(train_data['idx_train'])]
                    picked_test_data = train_data['idx_test'][int(self.id)%len(train_data['idx_test'])]
                    # print("picked_data: ", len(picked_train_data))

                    # test initial model
                    # model = choice_model(MODEL_NAME)
                    # model.load_state_dict(choice_model(MODEL_NAME).state_dict())
                    model = choose_model_partition(MODEL_PARTITION_NAME)
                    model.load_state_dict(choose_model_partition(MODEL_PARTITION_NAME).state_dict())

                    train_dataset, test_dataset = get_dataset(DATASET_NAME)

                    # acc = test_inference(model, test_dataset, picked_teat_data)
                    # accuracy = acc
                    # print("initial accuracy: ", accuracy)

                    start = time.time()*1000
                    print("%s start training ...." % self.id)
                    localModel = local_train(self, model, train_dataset, picked_train_data)

                    # start to train
                    # start = time.time()
                    # localModel = local_train(self, model, train_dataset, picked_train_data, is_flipping)
                    print('%s train time %s ' % (self.id, time.time()*1000 - start))
                print('+++ localModel type: ',  type(localModel))
                update = {
                    'type': 'update',
                    'model': localModel,
                    'worker_size': WORKERS,
                    'reward': reward,
                    'deadline': deadline,
                    'time': time.time()*1000
                }
                update_model = pickle.dumps(update)
                # workers send update back to the publisher
                self.send_to_node(node, update_model)
                # self.node_message(node, update_model)
            #  publisher receives updates 
            elif received_data['type'] == 'update':
                print("%s receving updated model from %s "% (self.id, node.id))
                print("+++ update from %s transmission time: %s"% (node.id, time.time()*1000 - float(received_data['time'])))
                update = received_data
                # print('~~~~~~~~~~~~~~: received_data: ', len(received_data))
                update_model = update['model']
                start_time = update['time']
                reward = update['reward']
                deadline = update['deadline']

                localModel = None
                # localModel = choice_model(MODEL_NAME)
                # localModel.load_state_dict(update_model)
                if MODEL_PARTITION_NAME == 'awd':
                    localModel = awd_model
                    # localModel.load_state_dict(update_model)
                elif MODEL_PARTITION_NAME == 'transformer':
                    localModel = getTransformerModel()
                    # localModel.load_state_dict(update_model)
                else:
                    localModel = choose_model_partition(MODEL_PARTITION_NAME)
                
                localModel.load_state_dict(update_model)

                end_time = time.time()*1000

                record[node.id] = str(start_time) + ',' + str(end_time) + ',' + reward + ',' + deadline

                workers_size = update['worker_size']
                # print('worker_size: ', workers_size)
                
                if MODEL_PARTITION_NAME == 'awd' or MODEL_PARTITION_NAME == 'transformer':
                    accepted_model[int(node.id)] = localModel
                else: 
                    accepted_model[int(node.id)] = copy.deepcopy(localModel)
                # print('accepted_model keys: ', accepted_model.keys())

                if len(accepted_model) == workers_size:
                    for i in range(self.peers):
                        if i+1 not in accepted_model:
                            accepted_model[i+1] = None
                    # print("*******accepted_model******** ", accepted_model.keys())
                    # comput krum score to give rewards
                    distances = []
                    if MODEL_PARTITION_NAME == 'awd':
                        distances = krum_distance(accepted_model, self.peers, awd_model)
                    elif MODEL_PARTITION_NAME == 'transformer':
                        distances = krum_distance(accepted_model, self.peers, getTransformerModel())
                    else:    
                        distances = krum_distance(accepted_model, self.peers)
                    _, scores = krum(workers_size,  workers_size//3, distances)
                    print('scores: ', scores)
                    
                    print('record: ', record)

                    localModel_set = []
                    if MODEL_PARTITION_NAME == 'awd':
                        localModel_set = multi_krum(accepted_model, workers_size, workers_size//3, self.peers, awd_model)
                    elif MODEL_PARTITION_NAME == 'transformer':
                        localModel_set = multi_krum(accepted_model, workers_size, workers_size//3, self.peers, getTransformerModel())
                    else:    
                        localModel_set = multi_krum(accepted_model, workers_size, workers_size//3, self.peers)
                    
                    # print('localModel_set ', localModel_set)

                    ''' worked with blockchain '''
                    # self.addRecord(record, localModel_set, scores, node)
                            
                    localModels = [accepted_model[i].state_dict() for i in localModel_set]
                    globalModel = FedAvg(localModels)
                    
                    print('globalModel: ', len(globalModel))
                    accepted_model.clear()
                    record.clear()

                    print('finished!')

                    # model_test = choice_model(MODEL_NAME)
                    # model_test.load_state_dict(globalModel)
                    # _, test_dataset = get_dataset(DATASET_NAME)

                    # acc = test_inference(model_test, test_dataset)
                    # accuracy = acc
                    # print("After accuracy: ", accuracy)

        # receive publish
        elif data["type"] == "msg_pub":
            print("server %s receive task: %s from  %s" % (self.id, data, node.id))
            publish_time = data["time"]
            print('publish_time: ',  publish_time)

            res = {
                "type": "msg_res", 
                "id": self.id,
                "time": publish_time,
                "reward": data['rewards'],
                "deadline": data['deadline']
            }
            # give response
            self.send_to_node(node, res)
        # publisher recieve responses
        elif data["type"] == "msg_res" :
            print("server %s receive responses from: %s" % (self.id, node.id))
            IDs.append(node.id)
            res_time = time.time()*1000

            # received respose dict, save delay : id time
            delays[node.id] = res_time - data['time']
            # start to choose
            # sleep(TIMEOUT)
            # reputations = {}
            # with open('repu.txt') as f:
            #     for line in f:
            #         id, repu = line.replace('\n','').split(':')
            #         if id in IDs:
            #             reputations[id] = repu
                
            expected_res = 2*self.peers // 3
            # expected_workers = self.peers // 2

            if sum(1 for ids in IDs) >= expected_res:
                received = IDs.copy()
                IDs.clear()
                res_delay = {}
                # print("delays: ", delays)
                for id in received:
                    if id in delays:
                        res_delay[id] = delays[id]
                delays.clear()
                # print("res_delay: ", res_delay)
                ''' worked with blockchain '''
                # reputations = self.getReputation(received)

                # mimic reputation
                reputations = {"1": "3.1", "2": "2.0", "3": "2.2", "4": "1.8", "5": "2.8", "6": "1.9", "7": "3.3", "8": "2.5", "9": "0.9", "10": "2.7"}
                scores = {}
                weight_a = 0.5
                weight_b = 0.5
                for k, v in res_delay.items():
                    scores[k] = weight_b*float(reputations[k])*10 - weight_a*v
                # sorted_reputations = dict(sorted(reputations.items(), key=operator.itemgetter(1)))
                # print("sorted_reputations: ", sorted_reputations)
                # cut_reputations = dict(itertools.islice(sorted_reputations.items(), WORKERS))
                # print("cut repu: ", cut_reputations)

                sorted_scores = dict(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))
                print("sorted_scores: ", sorted_scores)

                workers =  dict(itertools.islice(sorted_scores.items(), WORKERS))
                print('workers: ', workers)
                
                exclude = []
                for v in range(1, CANDIDATES):
                    if str(v) not in workers:
                        exclude.append(str(v))
                
                print("exclude: ", exclude)
                
                train_data = ['train_data', WORKERS, data['reward'], data['deadline']]
                # send train task to workers. go to 'isinstance(data, list)'
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
        
