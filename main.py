# library imports
import time
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# local imports
from data import *
from train import *
from model import *

if __name__ == '__main__':
    data = pd.read_csv('data.csv')

    nlp_eng = spacy.load("en_core_web_sm")
    nlp_esp = spacy.load("es_core_news_sm")
    english = Vocabulary(nlp_eng, specials=['<pad>', '<sos>', '<eos>', '<unk>'])
    spanish = Vocabulary(nlp_esp, specials=['<pad>', '<sos>', '<eos>', '<unk>'])

    # tokenize
    data['eng_tokens'] = [english.tokenize(s) for s in data['english']]
    data['esp_tokens'] = [spanish.tokenize(s) for s in data['spanish']]
    data['eng_num'] = data['eng_tokens'].apply(lambda arr: len(arr) + 2)
    data['esp_num'] = data['esp_tokens'].apply(lambda arr: len(arr) + 2)
    # sort by length for dynamic padding
    data = data.sort_values(by='eng_num')

    # build vocabs
    for sample in data['eng_tokens']:
        for t in sample:
            english.insert(t)

    for sample in data['esp_tokens']:
        for t in sample:
            spanish.insert(t)

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=1337)

    # hyperparams
    BATCH_SIZE = 32
    HIDDEN_SIZE = 512
    SRC_EMB = 256
    TRG_EMB = 256
    SRC_VOCAB = len(english)
    TRG_VOCAB = len(spanish)
    lr = 0.001
    epochs = 25

    # prep dataloaders
    train_batches = split_batches(english(train_data['eng_tokens']),
                                  spanish(train_data['esp_tokens']),
                                  BATCH_SIZE)
    train_dl = DataLoader(dataset=train_batches, collate_fn=collate)

    val_batches = split_batches(english(val_data['eng_tokens']),
                                spanish(val_data['esp_tokens']),
                                BATCH_SIZE)
    val_dl = DataLoader(dataset=val_batches, collate_fn=collate)

    # test live translation
    test = "the man helped me and my horse cross the river"

    # init model
    torch.manual_seed(1337)
    model = EncDec(HIDDEN_SIZE, SRC_EMB, SRC_VOCAB, TRG_EMB, TRG_VOCAB)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    prev_time = time.time()
    train_losses = []
    val_losses = []
    for e in range(1, epochs + 1):
        epoch_loss = train_epoch(train_dl, model, optim, loss_fn).item()
        train_losses.append(epoch_loss)

        val_loss = validate_epoch(val_dl, model, loss_fn).item()
        val_losses.append(val_loss)

        t = time.time()
        print(f"Epoch: {e} --- Loss: {epoch_loss} --- dt: {round((t - prev_time) / 60, 4)} min")
        prev_time = t

        print(translate_string(test, english, spanish, model))


    # plot data
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.xlabel('epochs')
    plt.show()