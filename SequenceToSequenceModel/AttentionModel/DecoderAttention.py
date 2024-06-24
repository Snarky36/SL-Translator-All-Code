import torch.nn as nn
import torch

class DecoderAttention(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, nr_layers, dropout, attention):
        super(DecoderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nr_layers = nr_layers
        self.attention = attention
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size + embedding_size, hidden_size, nr_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(attn_weights, encoder_outputs).permute(1, 0, 2)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        output = self.fc(torch.cat((output.squeeze(0), context.squeeze(0)), dim=1))
        return output, hidden, cell
