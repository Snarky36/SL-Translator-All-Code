import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from datasets import load_dataset
from Decoder import Decoder
from Encoder import Encoder
from Seq2SeqModel import Seq2Seq
from Utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def getDatasets():
    dataset_train = load_dataset(
        './dataset/',
        data_files='PHOENIX-2014-T.train.corpus2.csv',
        split='train'
    )

    dataset_test = load_dataset(
        './dataset/',
        data_files='PHOENIX-2014-T.test.corpus2.csv',
        split='train'
    )

    dataset_dev = load_dataset(
        './dataset/',
        data_files='PHOENIX-2014-T.dev.corpus2.csv',
        split='train'
    )

    return dataset_train, dataset_test, dataset_dev


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


### We're ready to define everything we need for training our Seq2Seq model ###

#Training HyperParams

learning_rate = 0.01
batch_size = 64

# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(len(german.vocab))
print(len(english.vocab))
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 400
decoder_embedding_size = 400
hidden_size = 1024  # Needs to be the same for both RNN's
num_layers = 4
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard to get nice loss plot

print("train_data", train_data.shape)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

writer = SummaryWriter(f"runs/loss_plot")


encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net, english.vocab).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


def trainModel(num_epochs=20):
    step = 0
    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        # checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        # save_checkpoint(checkpoint)
        #
        # model.eval()
        #
        # translated_sentence = translate_sentence(
        #     model, sentence, german, english, device, max_length=50
        # )
        #
        # print(f"Translated example sentence: \n {translated_sentence}")

        model.train()

        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            print(batch)
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # Forward prop
            output = model(inp_data, target)

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin. While we're at it
            # Let's also remove the start token while we're at it
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            # Back prop
            loss.backward()

            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # Plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1


    score = bleu(test_data[1:100], model, german, english, device)
    print(f"Bleu score {score*100:.2f}")

def transalate(sentence):

    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.eval()  # Set the model to evaluation mode
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    return translated_sentence

def blueScore():
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    return bleu(test_data, model, german, english, device)


trainModel()