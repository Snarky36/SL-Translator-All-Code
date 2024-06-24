import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from Utils import save_checkpoint, load_checkpoint, translate_sentence
from torch.utils.tensorboard import SummaryWriter
import spacy #library for tokenizing text
import random

from Decoder import Decoder
from Encoder import Encoder
from Seq2SeqModel import Seq2Seq
spacy_german = spacy.load("de_core_news_sm")
spacy_english = spacy.load("en_core_web_sm")

def german_tokenizer(text):
    return [token.text for token in spacy_german.tokenizer(text)]

def english_tokenizer(text):
    return [token.text for token in spacy_english.tokenizer(text)]

german = Field(tokenize=german_tokenizer, lower=True, init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=english_tokenizer, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

def show_germanField():
    print(german)

#Training Hyper parameters
num_epochs = 20
learning_rate = 0.01
batch_size = 64

#Model hyper parameters
load_model = False
device = torch.device('cuda')
print('Device is using cuda hopefully = ', device)
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)

output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

#TensorBoard
writer = SummaryWriter(f'runs/loss_plot')
step = 0
training_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_key=lambda x: len(x.src),
    sort_within_batch=True,
    device=device
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size,
                      num_layers, encoder_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size,
                      output_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_index = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=40)

    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

for batch_idx, batch in enumerate(training_iterator):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(input_data, target[:-1])  # Ensure target doesn't include the last token
        output_dim = output.shape[-1]

        output = output.view(-1, output_dim)
        target = target[1:].view(-1)  # Shift target to exclude the first token

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Check if the loss is not None before logging
        if loss is not None:
            writer.add_scalar('Training loss', loss.item(), global_step=step)

        step += 1


