import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((vocab_size, embedding_dim)))

    def forward(self, token_ids):
        return self.weight[token_ids]

# -------------------------------------------------------------------------------------------------------

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.randn((input_dim + hidden_dim, 4 * hidden_dim)) / (input_dim + hidden_dim)**0.5)
        self.b = nn.Parameter(torch.zeros(4 * hidden_dim))

    def forward(self, x_t, h_prev, c_prev):

        # concatenating the input and the previous hidden state
        combined = torch.cat((x_t, h_prev), dim=1)
        gates = combined @ self.W + self.b

        # splitting the gates into input, forget, output, and cell gates
        i_t, f_t, o_t, g_t = gates.chunk(4, dim=1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        # updating the cell state and hidden state
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

# -------------------------------------------------------------------------------------------------------

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):  # x shape: (batch, seq_len, input_dim)
        B, T, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_dim, device=device)
        c = torch.zeros_like(h)
        outputs = []

        for t in range(T):
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(h)

        return torch.stack(outputs, dim=1)  # shape: (B, T, hidden_dim)

# -------------------------------------------------------------------------------------------------------

class Linear(nn.Module):
    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((fan_in, fan_out)) / fan_in**0.5)
        self.bias = nn.Parameter(torch.zeros(fan_out)) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out