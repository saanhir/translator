import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hidden_size, embed_size, vocab_size):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, source):
        # source = (N, L)
        embed = self.embedding(source) # (N, L, Emb)
        encodings, _ = self.gru(embed) # (N, L, H)
        return encodings


class Decoder(nn.Module):
    def __init__(self, hidden_size, embed_size, vocab_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRUCell(2*hidden_size, hidden_size)
        self.lin = nn.Linear(hidden_size, vocab_size)

        self.attention = Attention(hidden_size)

    def forward(self, encodings, target=None, maxlen=20):
        if target is not None:
            target = self.embedding(target) # (N, L, Emb)

        batch, src_len, _ = encodings.shape
        state = encodings[:, src_len-1, :] # s0 = (N, H)

        out_len = target.shape[1] if target is not None else maxlen
        x = target[:,0,:] if target is not None else self.embedding(torch.tensor( [1] ))

        # <sos>
        outputs = self.init_outputs(batch, self.vocab_size)

        for i in range(1, out_len):
            state = self.step(state, encodings, x)
            pred = self.lin(state) # (N, V)
            choice = pred.argmax()
            outputs.append(pred)

            # quit early
            if target is None and choice == 2: # eos
                break

            # next input
            if target is not None:
                x = target[:, i, :]   # (N, Emb=H)
            else:
                x = self.embedding(torch.tensor( [choice]) ).view(batch, self.embed_size)

        return torch.stack(outputs).permute(1, 0, 2) # (N, L, V)


    def step(self, state, encodings, token):
        # generate next state
        weights = self.attention(state, encodings)
        context = torch.bmm(weights.unsqueeze(1), encodings).squeeze(1) # (N, H)

        x = torch.cat( [token, context], dim=1) # (N, 2H)
        state = self.gru(x, state)
        return state


    def init_outputs(self, bsize, vsize):
        outputs = []
        sos = torch.zeros(bsize, vsize)
        sos[:, 1] = 1
        outputs.append(sos)
        return outputs


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.lin_k = nn.Linear(hidden_size , hidden_size)
        self.lin_q = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query = (N,H) // keys = (N,L,H)
        query = query.unsqueeze(1)
        scores = self.out(torch.tanh(self.lin_q(query) + self.lin_k(keys))) # (N, L, 1)
        weights = F.softmax( scores.squeeze(2), dim=-1 ) # (N, L)
        return weights


class EncDec(nn.Module):
    def __init__(self, hidden_size, src_embed_size, src_vocab_size, trg_embed_size, trg_vocab_size):
        super().__init__()

        self.encoder = Encoder(hidden_size, src_embed_size, src_vocab_size)
        self.decoder = Decoder(hidden_size, trg_embed_size, trg_vocab_size)

    def forward(self, source, target=None, maxlen=20):
        encodings = self.encoder(source)
        decoding = self.decoder(encodings, target=target, maxlen=maxlen)

        if target is not None:
            return decoding
        else:
            return decoding.argmax(2).squeeze(0)