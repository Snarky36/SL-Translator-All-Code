import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, nr_layers, dropout):
        super(GRUEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nr_layers = nr_layers
        self.embedding_size = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(p=dropout)  # Randomly zeroes some of the elements of the input tensor with probability p
        self.rnn = nn.GRU(embedding_size, hidden_size, nr_layers, dropout=dropout)  # Using GRU instead of LSTM

    def forward(self, indexVector):
        # indexVector looks like (sequence length, batch size)

        # Embedding and applying dropout
        embedding = self.dropout(self.embedding_size(indexVector))  # embedding shape (seq_length, batch_size, embedding_size)
        # each word will have a vector of (embedding size) dimension

        # Pass the embeddings through the GRU
        results, hidden = self.rnn(embedding)  # GRU only returns hidden state, no cell state

        return hidden  # Only hidden state is returned, as GRU does not have cell state
