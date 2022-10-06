DATASET_NAME = 'cifar' #cifar100
## net (cnn)
# google
# alex
# mobile
# resnet50
# vgg16

# awd
# transformer
# MODEL_NAME = 'awd'
NODE_PATH = '/home/liang/.nvm/versions/node/v8.17.0/bin/node'
# ATTACK_USERS = ['2','8', '5']
# add_gaussian_noise, sign_flipping_attack, label_flipping_attack
# ATTACK_TYPE = 'label_flipping_attack'

# indicate required workers
WORKERS = 5
CANDIDATES = 10

# util choose_model_partition()
# resnet50l2_1
# alexl4_1
# awd
# transformer
MODEL_PARTITION_NAME = 'alexl4_1'