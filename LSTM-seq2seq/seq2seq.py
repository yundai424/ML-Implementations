import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from util import *
from config import get_config
import os

class Encoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(Encoder, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.f = nn.Linear(n_in + n_hid, n_hid)
        self.i = nn.Linear(n_in + n_hid, n_hid)
        self.c = nn.Linear(n_in + n_hid, n_hid)
        self.o = nn.Linear(n_in + n_hid, n_hid)
    
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
        return hidden, ctx
        
    def init_hidden(self):
        return torch.zeros(1, self.n_hid)

class Decoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(Decoder, self).__init__()
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
        output = self.fc(hidden)
        return output, (hidden, ctx)
        
    def init_hidden(self):
        return torch.zeros(1, self.n_hid)


def train(train_pair, encoder, decoder, embeddings, optimizer):
    # train the model with one pair of sentences
    s1, s2 = train_pair
    losses = torch.zeros_like(s1[0])
    accu = 0
    hidden, ctx = None, None
    for s in s1:
        hidden, ctx = encoder(s, hidden, ctx)
    for i in range(1, len(s2)):
        curr = s2[i - 1]
        nxt = s2[i]
        output, (hidden, ctx) = decoder(curr, hidden, ctx)
        losses += (output - nxt) ** 2
        if same_word(pred_word_vector(output, embeddings), nxt):
            accu += 1
    
    loss = torch.sum(losses) / len(s2 - 1)
    accu = accu / len(s2 - 1)
    optimizer.zero_grad()
    loss.backward()
    # gradient clipping
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
    optimizer.step()
    
    return loss.item(), accu

def predict(pred_pair, encoder, decoder, embeddings, all_vocabs, teach):
    pred_words = ['<s>']
    pred_vecs = [word_to_tensor('<s>', embeddings, all_vocabs)]
    s1, s2 = pred_pair
    accu = 0
    hidden, ctx = None, None
    for s in s1:
        hidden, ctx = encoder(s, hidden, ctx)
    if teach:
        for i in range(1, len(s2)):
            curr = s2[i - 1]
            nxt = s2[i]
            output, (hidden, ctx) = decoder(curr, hidden, ctx)
            pred_vecs.append(pred_word_vector(output, embeddings))
            pred_words.append(pred_word(output, embeddings, all_vocabs))
            if same_word(pred_word_vector(output, embeddings), nxt):
                accu += 1
    else:
        while True:
            output, (hidden, ctx) = decoder(pred_vecs[-1], hidden, ctx)
            pred_vecs.append(pred_word_vector(output, embeddings))
            pred_words.append(pred_word(output, embeddings, all_vocabs))
            if len(pred_words) > len(s2)+5 or pred_words[-1] in ['.', '?', '!', '</s>']:
                break
        # calculate accuracy
        for i in range(len(s2)):
            if i < len(pred_vecs) and same_word(pred_vecs[i], s2[i]):
                accu += 1
    return ' '.join(pred_words), accu / (len(s2) - 1)

def train_all(train_pairs, config, max_num, embeddings, temporal):
    encoder = Encoder(ndim, ndim, ndim)
    decoder = Decoder(ndim, ndim, ndim)
    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), 
                                lr=config.lr, momentum=0.9)
    max_epoch = config.max_epoch
    prev_loss = None
    all_losses = []
    print('Begin training...')
    for i in range(max_epoch):
        losses = 0
        accuracy = 0
        counter = 0

        for train_pair in train_pairs[:max_num]:
            loss, accu = train(train_pair, encoder, decoder, embeddings, optimizer)
            losses += loss
            accuracy += accu
            counter += 1

        avg_loss = losses / counter
        avg_accuracy = accuracy / counter
        all_losses.append(avg_loss)
        print('Epoch: {:6} | Loss: {:5.3f} | Acc: {:.3f}'.format(i, avg_loss, avg_accuracy))

        # learning rate decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * config.lr_decay

        if prev_loss is not None and abs(avg_loss - prev_loss) <= 0.01:
            break
        prev_loss = avg_loss

    if temporal:
        torch.save(encoder, './temp_encoder' + str(temporal) + '.pt')
        torch.save(decoder, './temp_decoder' + str(temporal) + '.pt')    
    else:
        torch.save(encoder, config.encoder_path)
        torch.save(decoder, config.decoder_path)
    print('model saved.')
    return all_losses

def predict_all(pairs, sentences, embeddings, all_vocabs, config, teach):
    """
    predict and save the predict results
    """
    encoder = torch.load(config.encoder_path)
    decoder = torch.load(config.decoder_path)

    pred_sentences = []
    accus = []
    for pair in pairs:
        pred_sentence, accu = predict(pair, encoder, decoder, embeddings, all_vocabs, teach)
        pred_sentences.append(pred_sentence)
        accus.append(accu)

    assert len(pred_sentences) == len(sentences)
    if not teach:
        with open(config.predict_result, 'w') as f:
            for i in range(len(sentences)):
                f.write(sentences[i][1] + '\t' + pred_sentences[i] + '\n')
    return np.mean(accus)

    


if __name__ == '__main__':
    config = get_config()
    all_vocabs, n_vocabs = get_all_vocabs(config)
    embeddings = get_embedding(config, all_vocabs)
    ndim = embeddings.size()[1]
    train_pairs, train_sentences = load_and_split(config.train_file, embeddings, all_vocabs)
    validate_pairs, validate_sentences = load_and_split(config.validate_file, embeddings, all_vocabs)
    test_pairs, test_sentences = load_and_split(config.test_file, embeddings, all_vocabs)

    train_loss = train_all(train_pairs, config, 3000, embeddings, False)
    plot_loss(train_loss, config)

    accu = predict_all(test_pairs, test_sentences, embeddings, all_vocabs, config, True)
    print('Test accuracy : {:.3f}'.format(accu))
    predict_all(test_pairs, test_sentences, embeddings, all_vocabs, config, False) # generate examples


    # ==== expetiments ====
    accus = []
    descriptions = ['random embedding', 'pre-computed embedding', 'lr=1', 'lr=0.5', 'lr=0.1', 'lr=0.05']
    counter = 1
    # embeddings
    for embed in ['random', 'precomp']:
        config_temp = get_config()
        config_temp.embedding = embed
        train_all(train_pairs, config, 200, embeddings, True)
        accus.append(predict_all(test_pairs, test_sentences, embeddings, all_vocabs, config, counter))
        counter += 1

    # lr
    for lr in [1, 0.5, 0.1, 0.05]:
        config_temp = get_config()
        config_temp.lr = lr
        train_all(train_pairs, config, 200, embeddings, True)
        accus.append(predict_all(test_pairs, test_sentences, embeddings, all_vocabs, config, counter))
        counter += 1
    print(accus)