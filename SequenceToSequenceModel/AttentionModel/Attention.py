import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
        self.v = nn.Parameter(torch.rand(decoder_hidden_size))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        seq_len = encoder_outputs.shape[0]

        # Repeat hidden state seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch size, seq_len, decoder_hidden_size]
        energy = energy.permute(0, 2, 1)  # [batch size, decoder_hidden_size, seq_len]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch size, 1, decoder_hidden_size]
        attention = torch.bmm(v, energy).squeeze(1)  # [batch size, seq_len]

        return F.softmax(attention, dim=1)
