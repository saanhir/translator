
def train_epoch(pipe, model, optim, f_loss):
    model.train()
    cum_loss = 0
    count = 0

    for sample in pipe:
        source, target = sample
        optim.zero_grad()

        y_hat = model(source, target)
        y_hat = y_hat.reshape(-1, y_hat.shape[-1]) # (N*L, V)
        y = target.reshape(-1) # (N*L)

        loss = f_loss(y_hat, y)
        loss.backward()
        optim.step()

        count += 1
        cum_loss += loss

    return cum_loss / count


def test_sentence(test, model, vocab, maxlen=20):
    model.eval()
    pred = model(test.unsqueeze(0), maxlen=maxlen)
    return ' '.join([vocab.get_itos()[t] for t in pred])