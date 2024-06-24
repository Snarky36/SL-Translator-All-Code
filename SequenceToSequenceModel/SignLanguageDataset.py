from datasets import load_dataset
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
import torch
from spacy.lang.de import German
nlp = German()
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SignLanguageDataset():
    def __init__(self, prefix, max_length=256, path="./dataset/"):
        self.prefix = prefix
        self.train_data, self.test_data, self.dev_data = self.get_dataset(path=path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        #self.tokenizer = nlp.tokenizer
        self.max_tokens = self.tokenizer.model_max_length
        self.max_input_length = max_length
        self.max_target_length = max_length

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        sample = self.train_data[idx]
        source_text = sample["text"]
        target_text = sample["target"]
        print(source_text ,target_text)
        source_tokens = self.tokenizer.tokenize(source_text)
        target_tokens = self.tokenizer.tokenize(target_text)
        return {
            "source_tokens": self.tokenizer.convert_tokens_to_ids(source_tokens),
            "target_tokens": self.tokenizer.convert_tokens_to_ids(target_tokens)
        }

    def get_dataset(self, path="./dataset/"):

        dataset_train = load_dataset(
            path=path,
            data_files='PHOENIX-2014-T.train.corpus2.csv',
            split='train'
        )

        dataset_test = load_dataset(
            path=path,
            data_files='PHOENIX-2014-T.test.corpus2.csv',
            split='train'
        )

        dataset_dev = load_dataset(
            path=path,
            data_files='PHOENIX-2014-T.dev.corpus2.csv',
            split='train'
        )

        return dataset_train, dataset_test, dataset_dev

    def preprocess_function(self, dataset):
        inputs = [self.prefix + ex for ex in dataset["text"]]

        targets = [ex for ex in dataset["target"]]
        model_inputs = self.tokenizer(text=inputs, text_target=targets, max_length=self.max_input_length,
                                      truncation=True, padding='max_length', add_special_tokens=True)
        return model_inputs

    def preprocess_function2(self, sentence):
        source_text = self.prefix + sentence["text"]
        target_text = sentence["target"]

        # Tokenize source and target texts
        source_tokens = self.tokenizer.tokenize(source_text)
        target_tokens = self.tokenizer.tokenize(target_text)

        print("Soruce tokens", source_tokens)

        # Truncate and pad tokenized sequences
        source_tokens = source_tokens[:self.max_input_length - 2]  # -2 to account for EOS and SOS tokens
        target_tokens = target_tokens[:self.max_target_length - 2]  # -2 to account for EOS and SOS tokens

        # Convert tokens to input IDs
        source_input_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        target_input_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        print("Source id", source_input_ids)
        print("vocab size for bert",self.tokenizer.vocab_size)
        # Add special tokens SOS, EOS, and PAD
        source_input_ids = [self.tokenizer.cls_token_id] + source_input_ids + [self.tokenizer.sep_token_id]
        target_input_ids = [self.tokenizer.cls_token_id] + target_input_ids + [self.tokenizer.sep_token_id]

        # Add PAD tokens to make both sequences equal length
        source_input_ids += [self.tokenizer.pad_token_id] * (self.max_input_length - len(source_input_ids))
        target_input_ids += [self.tokenizer.pad_token_id] * (self.max_target_length - len(target_input_ids))

        return {
            "input_ids": source_input_ids,
            "labels": target_input_ids
        }

    def get_tensor_datasets(self):

        tokenized_train_dataset = self.train_data.map(self.preprocess_function, batched=True)

        tokenized_test_dataset = self.test_data.map(self.preprocess_function, batched=True)

        tokenized_dev_dataset = self.dev_data.map(self.preprocess_function, batched=True)

        #print("[DataSet.get_tensor] input_ids = ",tokenized_train_dataset['input_ids'][0])
        #print("[DataSet.get_tensor] labels = ", tokenized_train_dataset['labels'][0])

        tokenized_train_dataset = TensorDataset(
            torch.tensor(tokenized_train_dataset['input_ids']),
            torch.tensor(tokenized_train_dataset['labels']),
        )

        tokenized_test_dataset = TensorDataset(
            torch.tensor(tokenized_test_dataset['input_ids']),
            torch.tensor(tokenized_test_dataset['labels']),
        )

        tokenized_dev_dataset = TensorDataset(
            torch.tensor(tokenized_dev_dataset['input_ids']),
            torch.tensor(tokenized_dev_dataset['labels']),
        )

        return tokenized_train_dataset, tokenized_test_dataset, tokenized_dev_dataset


    def get_tokenized_datasets(self):

        tokenized_train_dataset = self.train_data.map(self.preprocess_function, batched=True)

        tokenized_test_dataset = self.test_data.map(self.preprocess_function, batched=True)

        tokenized_dev_dataset = self.dev_data.map(self.preprocess_function, batched=True)

        return tokenized_train_dataset, tokenized_test_dataset, tokenized_dev_dataset

    def split_dataset_and_get_validation(self, dataset):
        dataset_size = len(dataset)
        validation_size = int(0.1 * dataset_size)
        train_size = dataset_size - validation_size
        train_dataset, validation_dataset = random_split(dataset,[train_size, validation_size])

        return train_dataset, validation_dataset


    def test(self):
        print(self.tokenizer(text="hello",text_target="hallo", max_length=10, truncation=True, padding='max_length'))

    def get_tokenizer(self):
        return self.tokenizer

    def get_train_vocab_size(self):

        return self.tokenizer.vocab_size

    def get_max_nr_of_tokens(self, respect_dataset_max_length=False):
        if respect_dataset_max_length:
            return self.max_input_length

        return self.max_tokens

    def test_tokens(self):
        #aici e problema la tokenizer ca scoate iduri uriase pentru unele cuvinte si eu nu imi setez encode si decoder sizeul destul de mare ca sa acopere
        #si tokenele cu iduri asa mari
        # Ce e de facut? Habar nu am ori sa schimb tokenizerul ori sa gasesc o modalitate sa aflu care e cel mai mare token id ori sa blochez token idurile la ceva mai mic
        # dar nu intelef de ce scoate tokeni asa mari? Asa multe cuvinte impartite sa fie? Nu prea cred ca am aprox 3000 de cuvinte diferite in tot datasetul

        #####

        #AM GASIT posibil daca bun vocab size -1 o sa fie ok incerc maine : self.tokenizer.vocab_size

        ####
        #tokenized_train_dataset = self.train_data.map(self.preprocess_function2, batched=False)

        self.preprocess_function2(self.train_data[1])

