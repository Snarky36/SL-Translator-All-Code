import re

import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


def translate_sentence(model, sentence, german, english, device, max_length=50):
    # print(sentence)

    # sys.exit()

    # Load german tokenizer
    spacy_ger = spacy.load("de_core_news_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # print(tokens)

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)
    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar", path="/Checkpoints"):
    final_path = path + filename
    print(f"=> Saving checkpoint at path {final_path}")
    torch.save(state, final_path)


def adjust_state_dict_keys(state_dict):
    adjusted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]
        else:
            new_key = key
        adjusted_state_dict[new_key] = value
    return adjusted_state_dict

def load_checkpoint(checkpoint, model, optimizer, use_attention=False, was_trained_with_multiple_gpus=False):
    adjusted_checkpoint = checkpoint.copy()  # Make a copy to avoid modifying the original checkpoint

    if was_trained_with_multiple_gpus:
        adjusted_checkpoint["state_dict"] = adjust_state_dict_keys(adjusted_checkpoint["state_dict"])

    # Adjust keys if needed
    if "encoder.embedding.weight" in adjusted_checkpoint["state_dict"]:
        adjusted_checkpoint["state_dict"]["encoder.embedding_size.weight"] = adjusted_checkpoint["state_dict"].pop(
            "encoder.embedding.weight")

    if "decoder.embedding.weight" in adjusted_checkpoint["state_dict"] and not use_attention:
        adjusted_checkpoint["state_dict"]["decoder.embedding_size.weight"] = adjusted_checkpoint["state_dict"].pop(
            "decoder.embedding.weight")

    model.load_state_dict(adjusted_checkpoint["state_dict"])
    optimizer.load_state_dict(adjusted_checkpoint["optimizer"])




def inference(model, tokenizer, sentence, max_length=50, use_attention=False):
    model.eval()
    device = model.get_device()

    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    print("Tokens", tokens)

    src_indexes = tokenizer.convert_tokens_to_ids(tokens)
    sentence_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        if use_attention:
            encoder_results, (hidden, cell) = model.encoder(sentence_tensor, use_attention=use_attention)
        else:
            hidden, cell = model.encoder(sentence_tensor)

    outputs = [tokenizer.convert_tokens_to_ids('[CLS]')]

    print("outputs", outputs)

    for i in range(max_length):
        previous_word_vector = torch.LongTensor([outputs[-1]]).to(device)
        #print("previous_word_vector", previous_word_vector)
        with torch.no_grad():
            if use_attention:
                output, hidden, cell = model.decoder(previous_word_vector, hidden, cell, encoder_results)
            else:
                output, hidden, cell = model.decoder(previous_word_vector, hidden, cell)

            best_guess = output.argmax(1).item()

        outputs.append(best_guess)
        if best_guess == tokenizer.convert_tokens_to_ids('[SEP]'):
            break

    translated_tokens = [tokenizer.convert_ids_to_tokens(i) for i in outputs]
    translated_tokens = translated_tokens[1:-1]
    # Scap de tokenul <sos>

    pattern = r'(?<=\w) (?=\w)' # regex sa scap de spatiile inutile

    translated_sentence = ' '.join(tokenizer.convert_tokens_to_string(translated_tokens))
    translated_sentence = re.sub(pattern, '', translated_sentence)

    return translated_sentence

# Example usage:
# translated_sentence = inference(model, tokenizer, "Your German sentence here.", device)
# print(translated_sentence)
