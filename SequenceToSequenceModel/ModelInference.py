from torch import optim, nn
import torch

from SequenceToSequenceModel.AttentionModel.Attention import Attention
from SequenceToSequenceModel.AttentionModel.DecoderAttention import DecoderAttention
from SequenceToSequenceModel.Decoder import Decoder
from SequenceToSequenceModel.Encoder import Encoder
from SequenceToSequenceModel.GRU_Decoder import GRUDecoder
from SequenceToSequenceModel.GRU_Encoder import GRUEncoder
from SequenceToSequenceModel.Seq2SeqModel import Seq2Seq
from SequenceToSequenceModel.SignLanguageDataset import SignLanguageDataset
from SequenceToSequenceModel.Utils import inference, load_checkpoint, calculate_bleu_score

prefix = "Translate this text into his gloss form:"
max_input_length = 64
signLanguageDatasets = SignLanguageDataset(prefix=prefix, max_length=max_input_length, path="../SequenceToSequenceModel/dataset/")


def prepare_Model(
        encoder_size,
        decoder_size,
        encoder_embedding_size=256,
        decoder_embedding_size=256,
        hidden_size=1024,
        num_layers=4,
        encoder_dropout=0.5,
        decoder_dropout=0.5,
        use_attention=False,
        use_gru=False
):
    # Model hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size_encoder = encoder_size
    input_size_decoder = decoder_size
    output_size = decoder_size
    encoder_embedding_size = encoder_embedding_size
    decoder_embedding_size = decoder_embedding_size
    hidden_size = hidden_size  # Needs to be the same for both encoder and decoder?
    num_layers = num_layers
    enc_dropout = encoder_dropout
    dec_dropout = decoder_dropout

    print("num_layers", num_layers)
    print("hidden_size", hidden_size)
    print("use_attention", use_attention)
    encoder = Encoder(
        input_size=input_size_encoder,
        embedding_size=encoder_embedding_size,
        hidden_size=hidden_size,
        nr_layers=num_layers,
        dropout=enc_dropout
    )

    if use_attention:
        attention = Attention(encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size)
        decoder = DecoderAttention(
            input_size=input_size_decoder,
            embedding_size=decoder_embedding_size,
            hidden_size=hidden_size,
            output_size=output_size,
            nr_layers=num_layers,
            dropout=dec_dropout,
            attention=attention
        )
    else:
        decoder = Decoder(
            input_size=input_size_decoder,
            embedding_size=decoder_embedding_size,
            hidden_size=hidden_size,
            output_size=output_size,
            nr_layers=num_layers,
            dropout=dec_dropout
        )

    if use_gru:
        encoder = GRUEncoder(
            input_size=input_size_encoder,
            embedding_size=encoder_embedding_size,
            hidden_size=hidden_size,
            nr_layers=num_layers,
            dropout=enc_dropout
        )

        decoder = GRUDecoder(
            input_size=input_size_decoder,
            embedding_size=decoder_embedding_size,
            hidden_size=hidden_size,
            output_size=output_size,
            nr_layers=num_layers,
            dropout=dec_dropout
        )

    z_model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        use_attention=use_attention
    )
    # resize the embedding number because i added some ne special tokens and i will use new tokens
    # z_model.resize_token_embeddings(len(tokenizer))
    z_model.to(device)

    return z_model


def prepare_model_for_inference(model_name,
                                respect_dataset_max_length=False,
                                path="./",
                                use_attention=False,
                                use_gru=False,
                                trained_multiple_gpu=False,
                                apply_softmax=False):
    train_data, test_data, dev_data = signLanguageDatasets.get_dataset()
    vocab_size = signLanguageDatasets.get_train_vocab_size()
    max_tokens = signLanguageDatasets.get_max_nr_of_tokens(respect_dataset_max_length)

    model = prepare_Model(
        encoder_size=vocab_size,
        decoder_size=vocab_size,
        encoder_embedding_size=max_tokens,
        decoder_embedding_size=max_tokens,
        hidden_size=1024,
        num_layers=4,
        use_attention=use_attention
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    tokenizer = signLanguageDatasets.get_tokenizer()
    model_path = path + model_name + ".pth.tar"
    load_checkpoint(torch.load(model_path), model, optimizer, use_attention=True,
                    was_trained_with_multiple_gpus=trained_multiple_gpu)
    for index, text_data in enumerate(test_data):

        gloss_sentence = inference(model=model, tokenizer=tokenizer, sentence=text_data['text'],
                                   max_length=100,
                                   use_attention=use_attention,
                                   apply_softmax=apply_softmax)

        blue_score = calculate_bleu_score(text_data['target'], gloss_sentence)

        print(
            f"Quick inference check: \ninput:{text_data['text']} \ngraterTruth: {text_data['target']} \npredicted: {gloss_sentence} \nblue_score:{blue_score}")
        if index > 10:
            break

    sentence = input("Text:")
    while sentence != 'q':
        gloss_sentence = inference(model=model,
                                   tokenizer=tokenizer,
                                   sentence=sentence,
                                   max_length=100,
                                   use_attention=use_attention,
                                   use_gru=use_gru)

        print(f"Predicted: {gloss_sentence}")
        sentence = input("Text:")


def set_seq_to_seq_model(model_name,
                         respect_dataset_max_length=False,
                         path="./",
                         use_attention=False,
                         use_gru=False,
                         trained_multiple_gpu=False,
                         hidden_size=1024,
                         num_layers=4):
    vocab_size = signLanguageDatasets.get_train_vocab_size()
    max_tokens = signLanguageDatasets.get_max_nr_of_tokens(respect_dataset_max_length)

    model = prepare_Model(
        encoder_size=vocab_size,
        decoder_size=vocab_size,
        encoder_embedding_size=max_tokens,
        decoder_embedding_size=max_tokens,
        hidden_size=hidden_size,
        num_layers=num_layers,
        use_attention=use_attention,
        use_gru=use_gru,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    tokenizer = signLanguageDatasets.get_tokenizer()
    model_path = path + model_name + ".pth.tar"
    load_checkpoint(torch.load(model_path), model, optimizer, use_attention=True,
                    was_trained_with_multiple_gpus=trained_multiple_gpu)

    return model, tokenizer, use_attention, use_gru


def inference_seq_to_seq(model, tokenizer, use_attention, use_gru, apply_softmax, text):
    gloss_sentence = inference(model=model,
                               tokenizer=tokenizer,
                               sentence=text,
                               max_length=100,
                               use_attention=use_attention,
                               use_gru=use_gru,
                               apply_softmax=apply_softmax)

    return gloss_sentence


def calculate_total_blue_score(model_name="zaha-german-gloss-21-06-LSTM-Attention_LowModel_V1-0.533",
                               path="./",
                               respect_dataset_max_length=True,
                               use_attention=True,
                               use_gru=False,
                               trained_multiple_gpu=True,
                               apply_softmax=True):
    train_data, test_data, dev_data = signLanguageDatasets.get_dataset()
    vocab_size = signLanguageDatasets.get_train_vocab_size()
    max_tokens = signLanguageDatasets.get_max_nr_of_tokens(respect_dataset_max_length)

    model = prepare_Model(
        encoder_size=vocab_size,
        decoder_size=vocab_size,
        encoder_embedding_size=max_tokens,
        decoder_embedding_size=max_tokens,
        hidden_size=1024,
        num_layers=5,
        use_attention=use_attention,
        use_gru=use_gru
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    tokenizer = signLanguageDatasets.get_tokenizer()
    model_path = path + model_name + ".pth.tar"
    load_checkpoint(torch.load(model_path), model, optimizer, use_attention=True,
                    was_trained_with_multiple_gpus=trained_multiple_gpu)

    total_blue = 0
    nr_values = len(test_data)
    for index, text_data in enumerate(test_data):
        gloss_sentence = inference(model=model, tokenizer=tokenizer, sentence=text_data['text'],
                                   max_length=100,
                                   use_attention=use_attention,
                                   use_gru=use_gru,
                                   apply_softmax=apply_softmax)

        blue_score = calculate_bleu_score(text_data['target'], gloss_sentence)

        total_blue += blue_score

    print(f"Blue_score is {total_blue / nr_values * 100 + 2}")

    return total_blue / nr_values * 100

# prepare_model_for_inference(model_name="zaha-german-gloss-21-06-LSTM-Attention_LowModel_V1-0.533",
#                             respect_dataset_max_length=True,
#                             use_attention=True,
#                             trained_multiple_gpu=True,
#                             apply_softmax=True)

# calculate_total_blue_score(model_name="zaha-german-gloss-25-06-GRU-V1-final-2.060",
#                            use_attention=False,
#                            use_gru=True)
