import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from util import *
from config import get_config
import os

class LSTMNet(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(LSTMNet, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.f = nn.Linear(n_in + n_hid, n_hid)
        self.i = nn.Linear(n_in + n_hid, n_hid)
        self.c = nn.Linear(n_in + n_hid, n_hid)
        self.o = nn.Linear(n_in + n_hid, n_hid)
        self.fc = nn.Linear(n_hid, n_out)
    
    def forward(self, x, hidden=None, ctx=None):
        if hidden is None:
            hidden = self.init_hidden()
        if ctx is None:
            ctx = self.init_hidden()
        cat_input = torch.cat((hidden, x), dim=1)
        f_t = torch.sigmoid(self.f(cat_input))
        i_t = torch.sigmoid(self.i(cat_input))
        c_t_temp = torch.tanh(self.c(cat_input))
        ctx = torch.mul(f_t, ctx) + torch.mul(i_t, c_t_temp)
        o_t = torch.sigmoid(self.o(cat_input))
        hidden = torch.mul(o_t, torch.tanh(ctx))
        out = self.fc(hidden)
        return out, (hidden, ctx)
        
    def init_hidden(self):
        return torch.zeros(1, self.n_hid)


def train(train_pair, model, embeddings, optimizer):
    # train the LSTM model with one pair of sentences
    s1, s2 = train_pair
    loss = torch.zeros_like(s1[0])
    accu = 0
    hidden, ctx = None, None
    for word_vec in s1:
        _, (hidden, ctx) = model(word_vec, hidden, ctx)
    for i in range(1, len(s2)):
        curr = s2[i - 1]
        nxt = s2[i]
        output, (hidden, ctx) = model(curr, hidden, ctx)
        loss += (output - nxt) ** 2
        pred_vec = pred_word_vector(output, embeddings)
        if torch.all(pred_vec == nxt).item():
            accu += 1
        
    loss = torch.sum(loss) / (len(s2) - 1)
    accu = accu / (len(s2) - 1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return hidden, loss.item(), accu


def run():
    config = get_config()
    all_vocabs, n_vocabs = get_all_vocabs(config)
    embeddings = get_embedding(config, all_vocabs)
    ndim = embeddings.size()[1]
    train_pairs = load_and_split(config.train_file, embeddings, all_vocabs)
    validate_pairs = load_and_split(config.validate_file, embeddings, all_vocabs)
    test_pairs = load_and_split(config.test_file, embeddings, all_vocabs)

    net = LSTMNet(ndim, ndim, ndim)
    optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)
    max_epoch = config.max_epoch
    prev_loss = None

    print('Begin trainning...')
    for i in range(max_epoch):
        losses = 0
        accuracy = 0
        counter = 0

        for train_pair in train_pairs:
            output, loss, accu = train(train_pair, net, embeddings, optimizer)
            losses += loss
            accuracy += accu
            pred_word_idx = get_nearest_idx(output, embeddings)
            counter += 1

        avg_loss = losses / counter
        avg_accuracy = accuracy / counter
        print('Epoch: {:6} | Loss: {:5.3f} | Acc: {:.3f}'.format(i, avg_loss, avg_accuracy))
        if prev_loss is not None and abs(avg_loss - prev_loss) <= 0.001:
            break
    torch.save(net, config.model_path)

if __name__ == '__main__':
    run()