from transformers import (T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoModelForSeq2SeqLM, pipeline)
from datasets import load_dataset, load_metric
import torch
import evaluate
import numpy as np

model_checkpoint = "GermanT5/t5-efficient-gc4-all-german-small-el32"
#model_checkpoint = "GermanT5/t5-efficient-gc4-all-german-large-nl36"
#model_checkpoint = "google/mt5-base"
# "t5-small"
# "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
text = "text"
target = "target"
prefix = "summarize: "
max_input_length = 256
max_target_length = 256


def testTokenizer():
    dataset_train = load_dataset(
        './custom_datasets/Pheonix/',
        data_files='PHOENIX-2014-T.train.corpus2.csv',
        split='train'
    )
    print("\n\nDataset:", dataset_train)

    target_datasetLower = [value.lower() for value in dataset_train[target]]
    print("lowered token: ", target_datasetLower[1])

    #tokenized_train_dataset = dataset_train.map(preprocess_function, batched=True)

    #print("Tokenized labels:", tokenized_train_dataset["text"][1])

    inputs = tokenizer(dataset_train[1][text])

    print("Inputs are = \n", inputs)

    wrong_tokens = tokenizer(target_datasetLower, truncation=True)
    print("\n\nWrong_tokens:", inputs)
    print(tokenizer.convert_ids_to_tokens(wrong_tokens["input_ids"][1]))
    print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

def testBlue():
    sacrebleu = load_metric("sacrebleu")
    predictions = ["hello there general kenobi", "foo bar foobar"]
    references = [["hello there general KENOBI", "hello there !"], ["foo bar foobar", "foo bar foobar"]]

    results = sacrebleu.compute(predictions=predictions, references=references, lowercase=True)
    print("Metric structure", sacrebleu)
    print("\n\n Results of blue score= ", list(results.keys()))
    print(round(results["score"], 1))

def testRouge():
    metric = evaluate.load("rouge")
    candidates = ["Summarization is cool","I love Machine Learning","Good night"]

    references = [["Summarization is beneficial and cool","Summarization saves time"],
            ["People are getting used to Machine Learning","I think i love Machine Learning"],
            ["Good night everyone!","Night!"]
                         ]
    results = metric.compute(predictions=candidates, references=references)
    print(results)
