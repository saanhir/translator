# library imports
import time
import spacy
import pandas as pd
from torch.utils.data import DataLoader

# local imports
from data import *
from train import *
from model import *

if __name__ == '__main__':
    data = pd.read_csv('data.csv')

    nlp_eng = spacy.load("en_core_web_sm")
    nlp_esp = spacy.load("es_core_news_sm")

    data['eng_tokens'] = [tokenize(s, nlp_eng) for s in data['english']]
    data['esp_tokens'] = [tokenize(s, nlp_eng) for s in data['spanish']]
    data['eng_num'] = data['eng_tokens'].apply(lambda arr: len(arr) + 2)
    data['esp_num'] = data['esp_tokens'].apply(lambda arr: len(arr) + 2)
    data = data.sort_values(by='eng_num')

    english = Vocabulary(specials=['<pad>', '<sos>', '<eos>', '<unk>'])
    spanish = Vocabulary(specials=['<pad>', '<sos>', '<eos>', '<unk>'])

    for sample in data['eng_tokens']:
        for t in sample:
            english.insert(t)

    for sample in data['esp_tokens']:
        for t in sample:
            spanish.insert(t)

    # hyperparams
    epochs = 25
    BATCH_SIZE = 16
    HIDDEN_SIZE = 128
    SRC_VOCAB = len(english)
    TRG_VOCAB = len(spanish)
    lr = 0.005

    # prep dataloader
    batches = split_batches(vocabularize_series(data['eng_tokens'], english),
                            vocabularize_series(data['esp_tokens'], spanish),
                            BATCH_SIZE)

    dl = DataLoader(dataset=batches, collate_fn=collate)

    # test live translation
    test = "<sos> the man helped me and my horse cross the river <eos>"
    test = torch.tensor([english.stoi[t] for t in test.split(' ')])

    # init model
    torch.manual_seed(1337)
    model = EncDec(HIDDEN_SIZE, HIDDEN_SIZE, SRC_VOCAB, HIDDEN_SIZE, TRG_VOCAB)
    f_loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    prev_time = time.time()
    for e in range(1, epochs + 1):
        epoch_loss = train_epoch(dl, model, optim, f_loss)

        t = time.time()
        print(f"Epoch: {e} --- Loss: {epoch_loss} --- dt: {round((t - prev_time) / 60, 4)} minutes")
        prev_time = t

        print(test_sentence(spanish, test, model))