from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoModelForSeq2SeqLM,
                          pipeline, TrainingArguments)

from datasets import load_dataset, load_metric
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from torch.optim import Adam
from peft import LoraConfig
import evaluate
import os
import numpy as np
import torch
import time
import sacrebleu

def postprocess_text(preds, labels):
    write_logs_to_file("\n\n[FinetuneV2] Preds: ");
    write_logs_to_file(preds)
    print("Preds: ", preds[1])
    print("\n\nLabels ", labels[1])
    write_logs_to_file("\n\n[FinetuneV2] Labels: ");
    write_logs_to_file(labels)
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

metrics_global_file_path="/out/Models_3_03_24/GermanGpt2_NEFT_Blue_v1/metrics.txt"
def calculate_bleu_score(predictions, references):
    """
    Calculate the BLEU score between a list of predictions and references.

    Args:
        predictions (List[str]): List of predicted sentences.
        references (List[str]): List of reference sentences.

    Returns:
        float: BLEU score.
    """
    # Tokenize references
    references_tokenized = [[ref.split()] for ref in references]

    # Tokenize predictions
    predictions_tokenized = [pred.split() for pred in predictions]

    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(predictions_tokenized, [references_tokenized])
    return bleu.score


def convert_and_write(logs):
    new_logs=""
    for s in logs:
        t = s.decode('iso-8859-1')
        new_logs += t.encode('utf-8')
    return new_logs

def write_logs_to_file(logs, file_path='/out/logs.txt'):
    with open(file_path, 'a') as file:
        try:
            file.write(str(logs))
        except NameError:
            print("Could not write some german characters:\n")
            print(NameError)
            file.write(str(convert_and_write(logs)))

class PhoenixData(Dataset):
    def __init__(self, path: str, tokenizer1):
        self.dataTrain, self.dataSet = getDatasets(path)
        self.trainList = []

        tokenizer1.add_special_tokens({"pad_token": "<pad>",
                                       "bos_token": "<sos>",
                                       "eos_token": "<eos>",
                                       "sep_token": "<Gloss>:"})
        #tokenizer1.add_tokens(["glossToken", "<Gloss>:"])

        for data in self.dataTrain:
            self.trainList.append(data)

        for index, data in enumerate(self.trainList):
            self.trainList[index] = "<sos> " + data["text"] + " <Gloss>: " + data["target"] + " <eos>"

        print(self.trainList[0:4])

        self.trainListEncodes = tokenizer1(self.trainList, truncation=True, padding=True, max_length=1024,
                                           return_tensors="pt").to("cuda")
        self.input_ids = self.trainListEncodes['input_ids'].to("cuda")
        self.attention_mask = self.trainListEncodes['attention_mask'].to("cuda")

    def __len__(self):
        return len(self.trainList)

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item]

    def getListEncoded(self):
        return self.trainListEncodes

    def getTestReferences(self):
        return self.dataSet["target"]

    def getTestText(self):
        return self.dataSet["text"]


decoder_model_checkpoint = "dbmdz/german-gpt2"
# decoder_model_checkpoint="xlm-clm-ende-1024"

tokenizer = AutoTokenizer.from_pretrained(decoder_model_checkpoint)
response_template = "\n###GlossForm: "
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
metrics_global_file_path = "/out/"

out_folder = "/out/Models_3_03_24/"

def getDatasets(path="./custom_datasets/Pheonix/"):
    dataset_train = load_dataset(
        path,
        data_files='PHOENIX-2014-T.train.corpus2.csv',
        split='train'
    )

    dataset_test = load_dataset(
        path,
        data_files='PHOENIX-2014-T.test.corpus2.csv',
        split='train'
    )

    return dataset_train, dataset_test


def formatting_prompts_func(dataset):
    output_texts = []
    input_prefix = "\n###Text: "
    output_prefix = "\n###GlossForm: "
    inputs = [input_prefix + ex for ex in dataset["text"]]
    outputs = [output_prefix + ex for ex in dataset["target"]]

    for i in range(len(inputs)):
        input_text = inputs[i]
        response = outputs[i]
        feedText = f'''Unten ist ein Text. Schreiben Sie die GlossForm dieses Textes.
                    {input_text}
                    {response}
                    '''
        output_texts.append(feedText)

    return output_texts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print("Manual Training")

    model = AutoModelForCausalLM.from_pretrained(decoder_model_checkpoint)
    dataset = PhoenixData("custom_datasets/Pheonix/", tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    trainingArgs = TrainingArguments(
        output_dir="/out/Models_3_03_24/GermanGpt2_NEFT_Blue_v1",
        warmup_steps=200,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-5,
        evaluation_strategy='steps',
        logging_dir="/out/",
        save_steps=1000,
        eval_steps=200,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=4,
        neftune_noise_alpha=5,
        fp16=True,
        dataloader_num_workers=4,
        seed=30,
    )
    metrics_global_file_path = "/out/Models_3_03_24/"
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     formatting_func=formatting_prompts_func,
    #     data_collator=collator,
    #     peft_config=peft_config,
    #     max_seq_length=1024,
    #     args=trainingArgs,
    # )
    model.to(device)
    model.train()
    batchedData = DataLoader(dataset, batch_size=4)
    optim = Adam(model.parameters(), lr=1e-5, weight_decay=0.5)
    last_bluescore = 0
    for epoch in tqdm(range(1)):
        step = 0
        predictions = []
        for trainList, attention in batchedData:
            step += 1
            optim.zero_grad()
            loss = model(trainList, attention_mask=attention, labels=trainList).loss
            print("Epoch = " + str(epoch) + " Step = " + str(step) + "\nloss" + str(loss))
            loss.backward()
            optim.step()

            if step % 3000 == 0:
                inf("und nun die wettervorhersage für morgen donnerstag den zwölften august", model, device)

        for textTest in dataset.getTestText():
            pred = inf(textTest, model, device)
            predictions.append(pred)

        current_BlueScore = calculate_bleu_score(predictions, dataset.getTestReferences())

        print("\nCurrent Blue Score", current_BlueScore)
        write_logs_to_file("\nBleuScore:", "/out/Models_3_03_24/GermanGpt2_NEFT_Blue_v1/metrics.txt")
        write_logs_to_file(str(current_BlueScore), "/out/Models_3_03_24/GermanGpt2_NEFT_Blue_v1/metrics.txt")

        if last_bluescore < current_BlueScore:
            last_bluescore = current_BlueScore
            torch.save(model.state_dict(), "/out/Models_3_03_24/model_trainedV2.pt")

    # trainer.train()
    #
    # save_model_path = metrics_global_file_path + "/finalModel"
    # trainer.save_model(save_model_path)


def inference():
    prefix = "Unten ist ein Text. Schreiben Sie die GlossForm dieses Textes.\n###Text: "

    # generate_kwargs = {"do_sample": True, "temperature": 0.0001, "max_new_tokens": 1024}

    model = pipeline("text-generation",
                     model="/out/Models_3_03_24/finalModel",
                     # max_new_tokens=500,
                     max_length=500,
                     tokenizer=tokenizer
                     )
    text = input("Input: ")
    while text != 'q':
        translate = prefix + text
        glosses = model(translate)
        print(glosses)
        text = input("Input: ")


def inf(text, model, device):
    translate = "<sos> " + text + " <Gloss>: "
    inputs = tokenizer(translate, return_tensors="pt")
    print(inputs)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():  # Disable gradient calculation during inference
        glosses_encoded = model.generate(input_ids,
                                         attention_mask=attention_mask,
                                         max_length=1024,
                                         pad_token_id=tokenizer.eos_token_id,
                                         num_return_sequences=1)



    glosses = tokenizer.decode(glosses_encoded[0], skip_special_tokens=True)  # Decode the generated tokens

    write_logs_to_file("\nText: ", "/out/Models_3_03_24/GermanGpt2_NEFT_Blue_v1/logs.txt")
    write_logs_to_file(translate, "/out/Models_3_03_24/GermanGpt2_NEFT_Blue_v1/logs.txt")
    write_logs_to_file("\nTruth: ", "/out/Models_3_03_24/GermanGpt2_NEFT_Blue_v1/logs.txt")
    write_logs_to_file("\nJETZT WETTER MORGEN DONNERSTAG ZWOELF FEBRUAR", "/out/Models_3_03_24/GermanGpt2_NEFT_Blue_v1/logs.txt")

    write_logs_to_file("\nGloss generate:\n", "/out/Models_3_03_24/GermanGpt2_NEFT_Blue_v1/logs.txt")
    write_logs_to_file(glosses, "/out/Models_3_03_24/GermanGpt2_NEFT_Blue_v1/logs.txt")
    print(glosses)
    return glosses

train()
