import torch
from torch import nn
from torch.autograd import Variable
import copy
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#from model.masked_cross_entropy import *

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

def calculate_coefficient(new_dataset, replay_dataset,tokenizer):
    vectorizer = CountVectorizer(min_df=10, ngram_range=(1,1))
    transformer = TfidfTransformer()
    new_train_corpus = [tokenizer.decode(i) for i in new_dataset['input_ids']]
    replay_train_corpus = [tokenizer.decode(i) for i in replay_dataset['input_ids']]
    train_tfidf = transformer.fit_transform(vectorizer.fit_transform(new_train_corpus+replay_train_corpus)).toarray().tolist()
    train_tfidf = np.array(train_tfidf)
    replay_feature = train_tfidf[:len(new_dataset)].mean(axis=0)
    new_feature = train_tfidf[len(new_dataset):].mean(axis=0)

    a = replay_feature.dot(new_feature)
    b = np.linalg.norm(replay_feature)
    c = np.linalg.norm(new_feature)
    return a/(b*c)

class EWC(object):
    def __init__(self, model, dataset, device, length):

        self.model = model
        # the data we use to compute fisher information of ewc (old_exemplars)
        self.dataset = dataset
        self.device = device
        self.length = length

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {} # previous parameters
        self._precision_matrices = self._diag_fisher() # approximated diagnal fisher information matrix

        for n, p in copy.deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):

        self.model.train()
        precision_matrices = {}
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        for batch in self.dataset:
            self.model.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                precision_matrices[n].data += p.grad.data ** 2 / self.length

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.model.zero_grad()
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        return loss