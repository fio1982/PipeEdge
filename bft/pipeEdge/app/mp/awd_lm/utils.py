import imp
from pyexpat import model
import torch
import os
import hashlib
from .splitcross import SplitCrossEntropyLoss
from .data import Corpus
import time
import numpy as np

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    # if args.cuda:
    #     data = data.cuda()
    return data


def get_batch(source, i, bptt, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


###############################################################################
# Load data
###############################################################################
def loadData(path):
    fn = 'corpus.{}.data'.format(hashlib.md5(path.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = Corpus(path)
        torch.save(corpus, fn)

    eval_batch_size = 10
    test_batch_size = 1
    train_batch_size = 80
    train_data = batchify(corpus.train, train_batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, test_batch_size)

    return train_data, corpus, val_data, test_data

def buildModel(corpus, model):
    criterion = None

    ntokens = len(corpus.dictionary)
    # model = model.RNNModel('args.model', ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    model = model.RNNModel('LSTM', ntokens, 400, 1150, 3, 0.4, 0.3, 0.65, 0.1, 0.5, True)

    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        # print('Using', splits)
        # emsize = 400
        criterion = SplitCrossEntropyLoss(400, splits=splits, verbose=False)
    ###
    cuda = False
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    ###
    params = list(model.parameters()) + list(criterion.parameters())
    # total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    # print('Model total parameters:', total_params)

    return model, criterion, params

def train(optimizer, train_data, model, criterion, params):
    # Turn on training mode which enables dropout.
    # if model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    # ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(80)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = 70 if np.random.random() < 0.95 else 70 / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / 70
        model.train()
        data, targets = get_batch(train_data, i, bptt, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        # output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        output = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        # loss = raw_loss
        # # Activiation Regularization
        # alpha = 2
        # if alpha: loss = loss + sum(alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # # Temporal Activation Regularization (slowness)
        # beta = 1
        # if beta: loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        # loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        clip = 0.25
        if clip: torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        
        batch += 1
        i += seq_len


def localtrain(choseModel, train_data, corpus, workerid):
    builtModel, criterion, params = buildModel(corpus, choseModel)
    # At any point you can hit Ctrl + C to break out of training early.
    model = builtModel
    optimizer = 'sgd'
    # learning rate
    lr = 30
    wdecay = 1.2e-6
    epochs = 10
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=wdecay)
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wdecay)
    start_time = time.time()*1000
    print('%s start training .....'% workerid)
    for epoch in range(1, epochs+1):
        train(optimizer, train_data, builtModel, criterion, params)
        print('| end of epoch {:3d} |'.format(epoch))

    print('| end of training | time: {:5.2f} ms'.format(time.time()*1000 - start_time))
        
    return builtModel.state_dict(), model