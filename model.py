import torch.nn as nn
from layers import Embedding, LSTM, Linear

class MovieTitleGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_dim)
        self.fc = Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
    def paramaters(self):
        return (self.embedding.parameters() + self.lstm.parameters() + self.fc.parameters())