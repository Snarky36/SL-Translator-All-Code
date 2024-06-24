import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, nr_layers, dropout):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.nr_layers = nr_layers
        self.embedding_size = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(p=dropout) #During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.
        self.rnn = nn.LSTM(embedding_size, hidden_size, nr_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size) #all the cells ar fully connected and applies a linear function to the input and result the output

    def forward(self, indexVector, hidden, cell):
        indexVector = indexVector.unsqueeze(0)
        #index vector looks like (1 , batch size) - we want the decoder to predict one word at a time
        #print("[Decoder] indexVector", indexVector.shape, " => ", indexVector)
        embedding = self.dropout(self.embedding_size(indexVector)) #embedding shape (1, batchsize, embddingsize)
        #each word will have a vector of (embedding size) dimension

        results, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        #results will look like (1, batch size, hidden_size)

        predictions = self.fc(results.squeeze(0)) #we just get rid of the first dimension

        return predictions, hidden, cell #return the prediction for the next word, the value of hidden layer and the previous word generated.
