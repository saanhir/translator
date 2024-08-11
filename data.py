import torch
import torch.nn.functional as F
import pandas as pd
import spacy


class Vocabulary:
    def __init__(self, spacy_nlp, specials=[]):
        self.size = 0
        self.stoi = {}
        self.itos = {}
        self.nlp = spacy_nlp

        for s in specials:
            self.insert(s)

    def insert(self, token):
        if token not in self.stoi:
            self.stoi[token] = self.size
            self.itos[self.size] = token
            self.size += 1

    def tokenize(self, string):
        doc = self.nlp(string)
        return [tok.text.lower() for tok in doc]

    def __len__(self):
        return self.size

    def __call__(self, series):
        return [torch.tensor([1] + [self.stoi[t] for t in row] + [2]) for row in series]


def split_batches(series1, series2, batch_size):
    batches = []
    for i in range(len(series1) // batch_size):
        batches.append(
            (series1[i * batch_size: (i + 1) * batch_size],
             series2[i * batch_size: (i + 1) * batch_size])
        )
    return batches


def collate(batch):
    batch = batch[0]
    src, trg = batch

    src_longest = max([len(x) for x in src])
    srcs = [F.pad(x, (0, src_longest - len(x)), value=0) for x in src]

    trg_longest = max([len(x) for x in trg])
    trgs = [F.pad(x, (0, trg_longest - len(x)), value=0) for x in trg]
    return torch.stack(srcs), torch.stack(trgs)
