from transformers import (T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer,AutoModelForCausalLM, TrainingArguments, Trainer,
                          Seq2SeqTrainingArguments, DataCollatorForLanguageModeling, Seq2SeqTrainer, AutoModelForSeq2SeqLM, pipeline)

#import TestUtils
from datasets import load_dataset, load_metric
from trl import SFTTrainer
from peft import LoraConfig
import evaluate
import os
import numpy as np

decoder_model_checkpoint = "dbmdz/german-gpt2"

tokenizer = AutoTokenizer.from_pretrained(decoder_model_checkpoint)

model = AutoModelForCausalLM.from_pretrained(decoder_model_checkpoint)

dataCollator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
save_model_path="/out/Models_3_03_24/"


def postprocess_text(preds, labels):
    # write_logs_to_file("\n\n[FinetuneV2] Preds: ");
    # write_logs_to_file(preds)
    print("Preds: ", preds[1])
    print("\n\nLabels ", labels[1])
    # write_logs_to_file("\n\n[FinetuneV2] Labels: ");
    # write_logs_to_file(labels)
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metricBlue(eval_pred):
    metric = load_metric("sacrebleu")
    preds, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    #write_logs_to_file("Started Metrics for model = ", file_path='/out/metriclogs.txt')
    #write_logs_to_file(model_checkpoint, file_path='/out/metriclogs.txt')
    #write_logs_to_file("\n\nBlue Score= ", file_path=metrics_global_file_path)
    #write_logs_to_file(result["score"], file_path=metrics_global_file_path)


    result = {"bleu": result["score"]}


    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
def getDatasets():
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

    return dataset_train, dataset_test


def trainer():

    train_dataset, test_dataset = getDatasets()
    training_args = TrainingArguments(
        output_dir="./gpt2-gerchef",  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=4,  # batch size for training
        per_device_eval_batch_size=4,  # batch size for evaluation
        eval_steps=400,  # Number of update steps between two evaluations.
        save_steps=800,  # after # steps model is saved
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        learning_rate=1e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=dataCollator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metricBlue,
    )

    trainer.train()

    trainer.save_model(save_model_path)

trainer()