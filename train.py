import torch

# TODO: cast samples to device
def train_epoch(dataloader, model, optim, f_loss):
    model.train()
    cum_loss = 0

    for sample in dataloader:
        source, target = sample
        source = source
        target = target
        optim.zero_grad()

        y_hat = model(source, target)
        y_hat = y_hat.reshape(-1, y_hat.shape[-1]) # (N*L, V)
        y = target.reshape(-1) # (N*L)

        loss = f_loss(y_hat, y)
        loss.backward()
        optim.step()

        cum_loss += loss

    return cum_loss / len(dataloader)


def validate_epoch(dataloader, model, f_loss):
    model.eval()
    cum_loss = 0

    with torch.no_grad():
        for sample in dataloader:
            source, target = sample
            source = source
            target = target

            y_hat = model(source, target)
            y_hat = y_hat.reshape(-1, y_hat.shape[-1])  # (N*L, V)
            y = target.reshape(-1)  # (N*L)

            loss = f_loss(y_hat, y)

            cum_loss += loss

    return cum_loss / len(dataloader)


def translate_string(string, src_vocab, trg_vocab, model):
    model.eval()
    string = torch.tensor([1] + [src_vocab.stoi[t] for t in src_vocab.tokenize(string)] + [2])
    maxlen = int(len(string) * 1.5 + 2)

    pred = model(string.unsqueeze(0), maxlen=maxlen) #todo: cast string to device
    return ' '.join([trg_vocab.itos[t.item()] for t in pred])