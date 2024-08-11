import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, hidden_size, embed_size, vocab_size, drop=0.1):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(drop)

    def forward(self, source):
        # source = (N, L)
        embed = self.dropout(self.embedding(source)) # (N, L, Emb)
        encodings, _ = self.gru(embed) # (N, L, H)
        return encodings



class Decoder(nn.Module):
    def __init__(self, hidden_size, embed_size, vocab_size, drop=0.1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRUCell(hidden_size + embed_size, hidden_size)
        self.lin = nn.Linear(hidden_size, vocab_size)

        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(drop)


    def forward(self, encodings, target=None, maxlen=20):
        if target is not None:
            target = self.embedding(target) # (N, L, Emb)

        batch, src_len, _ = encodings.shape
        state = encodings[:, src_len-1, :] # s0 = (N, H)

        out_len = target.shape[1] if target is not None else maxlen
        x = target[:,0,:] if target is not None else self.embedding(torch.tensor( [1] )) #todo: cast 1 to device

        # <sos>
        #outputs = torch.zeros(batch, self.vocab_size)
        outputs = self.init_outputs(batch, self.vocab_size)

        for i in range(1, out_len):
            state = self.step(state, encodings, self.dropout(x))
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
                x = self.embedding( choice ).view(batch, self.embed_size) #todo: cast choice to device

        return torch.stack(outputs).permute(1, 0, 2) # (N, L, V)


    def step(self, state, encodings, token):
        # generate next state
        weights = self.attention(state, encodings)
        context = torch.bmm(weights.unsqueeze(1), encodings).squeeze(1) # (N, H)

        x = torch.cat( [token, context], dim=1) # (N, H+Emb)
        state = self.gru(x, state)
        return state


    def init_outputs(self, bsize, vsize):
        outputs = []
        sos = torch.zeros(bsize, vsize) # TODO: cast to device
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
    def __init__(self, hidden_size, src_embed_size, src_vocab_size, trg_embed_size, trg_vocab_size, drop=0.1):
        super().__init__()

        self.encoder = Encoder(hidden_size, src_embed_size, src_vocab_size, drop=drop)
        self.decoder = Decoder(hidden_size, trg_embed_size, trg_vocab_size, drop=drop)

    def forward(self, source, target=None, maxlen=20):
        encodings = self.encoder(source)
        decoding = self.decoder(encodings, target=target, maxlen=maxlen)

        if target is not None:
            return decoding
        else:
            return decoding.argmax(2).squeeze(0)