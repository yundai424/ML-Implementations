import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

def get_embedding(config, all_vocabs):
    if config.embedding == 'random':
        res = random_embedding(all_vocabs)
    elif config.embedding == 'precomp':
        res = precomp_embedding(all_vocabs, config)
    elif config.embedding == 'onehot':
        res = onehot_embedding(all_vocabs)
    else:
        raise ValueError('Invalid embedding type!')
    print('Successfully generate word embedding.')
    return res

def get_all_vocabs(config):
    all_vocabs = []
    with open(config.all_voc_dict, 'r') as f:
        all_vocabs = f.read().splitlines()
    all_vocabs = list(set([s.lower() for s in all_vocabs]))
    return all_vocabs, len(all_vocabs)

def random_embedding(all_vocabs):
    # generate random word embedding
    random_dist = torch.distributions.uniform.Uniform(torch.Tensor([-1.0]), torch.Tensor([1.0]))
    return random_dist.sample(torch.Size([len(all_vocabs), 200]))

def precomp_embedding(all_vocabs, config):
    # if prepared np matrix file exists, just load it and return
    if os.path.exists(config.embedding_dict):
        embedding_matrix = np.load(config.embedding_dict)
    else:
        # load and save the embedding matrix
        embedding_matrix = np.zeros((len(all_vocabs), 200))
        word2idx = {}
        idx = 0
        vectors = []
        with open(config.glove_dict, 'rb') as f:
            for line in f:
                line = line.decode().split()
                word = line[0]
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)

        for idx, word in enumerate(all_vocabs):
            try:
                embedding_matrix[idx] = vectors[word2idx[word]]
            except KeyError:
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(200, ))
        np.save(config.embedding_dict, embedding_matrix)
    return torch.Tensor(embedding_matrix).unsqueeze(dim=2)

def onehot_embedding(all_vocabs):
    ndim = len(all_vocabs)
    embedding_matrix = torch.zeros(ndim, ndim, 1)
    for i in range(ndim):
        embedding_matrix[i][i][0] = 1
    return embedding_matrix

def word_to_idx(word, all_vocabs):
    # return the index of a word
    return all_vocabs.index(word)

def word_to_tensor(word, embeddings, all_vocabs):
    # transfer a string of word to given embedding space
    return torch.t(embeddings[word_to_idx(word.lower(), all_vocabs)])

def sentence_to_tensor(sentence, embeddings, all_vocabs):
    # tensor format: seq * batch * embedding size
    words = sentence.strip().split()
    tens = torch.zeros(len(words), 1, embeddings.size()[1])
    for idx, word in enumerate(words):
        tens[idx][0] = word_to_tensor(word.lower(), embeddings, all_vocabs)
    return tens

def load_and_split(filename, embeddings, all_vocabs):
    res = []
    sentences = []
    with open(filename, 'r') as f:
        all_pairs = f.read().splitlines()
    for pair in all_pairs:
        seq1, seq2 = pair.split('\t')
        sentences.append((seq1, seq2))
        res.append((sentence_to_tensor(seq1, embeddings, all_vocabs), 
                    sentence_to_tensor(seq2, embeddings, all_vocabs)))
    return res, sentences

def get_nearest_idx(output, embeddings):
    # get the embedding word vector which is closest to the given output
    similarities = F.cosine_similarity(embeddings.squeeze(dim=2), output)
    return torch.argmax(similarities).item()

def pred_word_vector(output, embeddings):
    # return the embedding vector of predicted word according to cosine similarity
    return torch.t(embeddings[get_nearest_idx(output, embeddings)])

def pred_word(output, embeddings, all_vocabs):
    # predict the word according to cosine similarity
    return all_vocabs[get_nearest_idx(output, embeddings)]

def plot_loss(losses, config):
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(config.loss_plot_path)

def same_word(vec1, vec2):
    return F.cosine_similarity(vec1, vec2, dim=1).item() > 0.95