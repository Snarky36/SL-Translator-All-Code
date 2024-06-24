from transformers import (T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq, Trainer)
from datasets import load_dataset, load_metric
import evaluate
import torch
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

#model_checkpoint = "GermanT5/t5-efficient-gc4-all-german-small-el32"
#model_checkpoint = "GermanT5/t5-efficient-gc4-all-german-large-nl36"
model_checkpoint = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
prefix = "translate: "
max_input_length = 128
max_target_length = 128
source_lang = "text"
target_lang = "target"
rouge = evaluate.load("rouge")

def write_logs_to_file(logs, file_path='/out/logs.txt'):
    with open(file_path, 'a') as file:
        file.write(str(logs))


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    write_logs_to_file("Pred[0]\n\n")
    write_logs_to_file(eval_preds[0])
    write_logs_to_file("\n\nPred[1] Lables\n\n")
    write_logs_to_file(eval_preds[1])
    preds, labels = eval_preds
    write_logs_to_file("\n\nPreds before transformation:\n")
    write_logs_to_file(preds)
    print("\n\nPreds before separation = \n", preds)
    if isinstance(preds, tuple):
        preds = preds[0]
    write_logs_to_file("\n\nPreds after the transformation:\n")
    write_logs_to_file(preds[0])
    print("\n\nPreds = \n\n", preds[0])
    #decoded_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in preds]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print("A ajuns pana aici 1")
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    print("A ajuns pana aici 2")
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    print("A ajuns pana aici 3!")
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def compute_metricsRouge(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def preprocess_function(examples):
    inputs = [prefix + ex for ex in examples[source_lang]]
    targets = [ex for ex in examples[target_lang]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='max_length')
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


Batch_Size = 8
def startPretrain(tokenized_train_dataset, tokenized_test_dataset, data_collator):
    training_args = Seq2SeqTrainingArguments(
        output_dir="/out/",
        num_train_epochs=2,
        per_device_train_batch_size=Batch_Size,
        per_device_eval_batch_size=Batch_Size,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="/out/",
        logging_steps=10,
        evaluation_strategy='steps',
        save_steps=20,
        predict_with_generate=True,
        eval_steps=20,
        load_best_model_at_end=True,
        save_total_limit=10,
        report_to='tensorboard',
        learning_rate=0.0002,
        fp16=True,
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metricsRouge,
        data_collator=data_collator
    )
    torch.cuda.empty_cache()
    history = trainer.train()

    write_logs_to_file("History\n\n" + history)

    print("Finetunning finished! Habar nu am daca a functionat sau nu! E timpul sa aflam")


if __name__ == '__main__':
    metric = load_metric("sacrebleu")
    # print(metric)

    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset_train = load_dataset(
        '/custom_datasets/Pheonix/',
        data_files='PHOENIX-2014-T.train.corpus2.csv',
        split='train'
    )
    dataset_test = load_dataset(
        '/custom_datasets/Pheonix/',
        data_files='PHOENIX-2014-T.test.corpus2.csv',
        split='train'
    )

    tokenized_train_dataset = dataset_train.map(preprocess_function, batched=True)
    tokenized_test_dataset = dataset_test.map(preprocess_function, batched=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    print("ZAHAND - Starting finetunning for T5 model on text to gloss translation...")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_checkpoint)

    startPretrain(tokenized_train_dataset, tokenized_test_dataset, data_collator)



