import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, nr_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nr_layers = nr_layers
        self.embedding_size = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(p=dropout) #During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.
        self.rnn = nn.LSTM(embedding_size, hidden_size, nr_layers, dropout=dropout)

    def forward(self, indexVector, use_attention=False):
        #index vector looks like (sequence length all the words in a sentence, batch size)

        #print("[Encoder] indexVector", indexVector.shape, " => ", indexVector)

        embedding = self.dropout(self.embedding_size(indexVector)) #embedding shape (seq_length, batchsize, embddingsize)
        #each word will have a vector of (embedding size) dimension
        results, (hidden, cell) = self.rnn(embedding)

        if use_attention:
            return results, (hidden, cell)

        return hidden, cell
