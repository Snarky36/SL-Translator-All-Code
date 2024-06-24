import torch.nn as nn

class GRUDecoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, nr_layers, dropout):
        super(GRUDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.nr_layers = nr_layers
        self.embedding_size = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(p=dropout)  # Randomly zeroes some of the elements of the input tensor with probability p
        self.rnn = nn.GRU(embedding_size, hidden_size, nr_layers, dropout=dropout)  # Using GRU instead of LSTM
        self.fc = nn.Linear(hidden_size, output_size)  # Applies a linear function to the input and result the output

    def forward(self, indexVector, hidden):
        indexVector = indexVector.unsqueeze(0)
        # index vector looks like (1, batch size) - we want the decoder to predict one word at a time
        embedding = self.dropout(self.embedding_size(indexVector))  # embedding shape (1, batch_size, embedding_size)
        # each word will have a vector of (embedding size) dimension

        results, hidden = self.rnn(embedding, hidden)
        # results will look like (1, batch size, hidden_size)

        predictions = self.fc(results.squeeze(0))  # we just get rid of the first dimension

        return predictions, hidden  # return the prediction for the next word and the value of hidden layer
