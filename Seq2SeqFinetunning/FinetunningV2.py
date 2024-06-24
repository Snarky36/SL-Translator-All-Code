from transformers import (T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration, AutoTokenizer,
                          AutoModelForCausalLM, GenerationConfig,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoModelForSeq2SeqLM,
                          pipeline)

from datasets import load_dataset, load_metric
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from sklearn.model_selection import train_test_split
import evaluate
import os
import torch
import numpy as np
from LogManager import LogManager
from MetricEvaluator import MetricEvaluator
import pandas as pd

#model_checkpoint = "deutsche-telekom/mt5-small-sum-de-mit-v1"
# model_checkpoint = "GermanT5/t5-efficient-gc4-all-german-small-el32"
# model_checkpoint = "GermanT5/t5-efficient-gc4-all-german-large-nl36"
# model_checkpoint = "google/mt5-base"
model_checkpoint = "google/mt5-large"
decoder_model_checkpoint = "xlm-clm-ende-1024"
# "t5-small"
# "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
text = "text"
target = "target"
prefix = "Translate this German text into his gloss form: "
max_input_length = 256
max_target_length = 256
rouge = evaluate.load("rouge")

global_file_path = '/out/'
global_models_folder = "Models_15_04_24"
global_model_name = "T5_GermanMT5Model_Enhanced_turbo"
log_manager = LogManager(global_file_path, global_models_folder, global_model_name, model_checkpoint)
metric_eval = MetricEvaluator(tokenizer, log_manager)


def convert_and_write(logs):
    new_logs = ""
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


def postprocess_text(preds, labels):
    log_manager.write_training_logs("\n\n[FinetuneV2] Preds: ")
    log_manager.write_training_logs(preds)
    print("Preds: ", preds[1])
    print("\n\nLabels ", labels[1])
    log_manager.write_training_logs("\n\n[FinetuneV2] Labels: ")
    log_manager.write_training_logs(labels)

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    return logits.argmax(dim=-1)


metrics_global_file_path = '/out/metriclogs.txt'


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

    # write_logs_to_file("Started Metrics for model = ", file_path='/out/metriclogs.txt')
    # write_logs_to_file(model_checkpoint, file_path='/out/metriclogs.txt')
    log_manager.write_metrics("\n\nBlue Score= ")
    log_manager.write_metrics(result["score"])

    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def compute_metricRouge(eval_pred):
    print("EVAL_PREDS: ", eval_pred)
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print("\n\nDecoded_Labels = ", decoded_labels[1])
    print("\n\nPredictions = ", decoded_preds)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    print("\n\n Rouge score", result)

    log_manager.write_metrics("\nRouge scroe:")
    log_manager.write_metrics(str(result))

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def setUpModel():
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
    return model


def translateGermanToEnglish(sentence):
    model = setUpModel()

    input_ids = tokenizer("Hello my name is Andrei",
                          return_tensors="pt").input_ids

    generatedIds = model.generate(input_ids=input_ids)
    print("Ids: ", generatedIds)
    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generatedIds
    ]

    translation = "".join(preds)

    print("Translation: ", translation)


def preprocess_function(dataset):
    inputs = [prefix + ex for ex in dataset[text]]
    targets = [ex for ex in dataset[target]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='max_length')

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def datasetInfo(dataset="german", load_enhanced_dataset=False):
    global prefix
    if(dataset == "american"):
        prefix = "Translate this English text into his gloss form: "
        american_dataset = load_dataset("aslg_pc12", split="train")
        american_dataset = american_dataset.rename_column('gloss', 'target')
        splitedData = american_dataset.train_test_split(test_size=0.35)
        dataset_train = splitedData["train"]
        dataset_eval_test = splitedData["test"].train_test_split(test_size=0.33)
        dataset_test = dataset_eval_test["train"]
        dataset_eval = dataset_eval_test["test"]

    else:
        if load_enhanced_dataset:
            dataset_train = load_dataset(
            '/custom_datasets/Pheonix/',
                data_files='PHOENIX-2014-T.train-enhanced.csv',
                split='train'
            )
            print("Enhanced size:", len(dataset_train))
        else:
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

        dataset_eval = load_dataset(
            '/custom_datasets/Pheonix/',
            data_files='PHOENIX-2014-T.dev.corpus2.csv',
            split='train'
        )

    tokenized_train_dataset = dataset_train.map(preprocess_function, batched=True)

    tokenized_test_dataset = dataset_test.map(preprocess_function, batched=True)

    tokenized_eval_dataset = dataset_eval.map(preprocess_function, batched=True)

    return tokenized_train_dataset, tokenized_test_dataset, tokenized_eval_dataset


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


def formatting_function(dataset):
    log_manager.write_errors("[FormattingFunction] Start\n")
    output_texts = []
    input_prefix = "\n###Text: "
    output_prefix = "\n###GlossForm: "
    inputs = [input_prefix + ex for ex in dataset[text]]
    outputs = [output_prefix + ex for ex in dataset[target]]

    for i in range(len(inputs)):
        input_text = inputs[i]
        response = outputs[i]
        feedText = f'''Unten ist ein Text. Schreiben Sie die GlossForm dieses Textes.
                    {input_text}
                    {response}
                    '''
        output_texts.append(feedText)

    log_manager.write_errors("[FormattingFunction] OutputsDone\n")
    return output_texts


def prepareTariningCollector():
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_checkpoint)
    return data_collator


def setTokenizerParalelism(state):
    os.environ["TOKENIZERS_PARALLELISM"] = state

def trainModel_Seq2SeqTrainer():
    print("Training Batch size: ")
    Batch_Size = int(input())
    print("\nTraining epochs: ")
    Num_Training_Epochs = int(input())
    dataset_language = input("Choose dataset (german/american): ")
    if dataset_language == "german":
        enhanced_dataset = bool(input("Choose enhanced dataset (True/False)"))
    print(
        "\nChoose the optimizer:none(will default the adamw_torch), adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor.\n")
    optimizer = input("Optimizer: ")
    if optimizer == "none":
        optimizer = "adamw_torch"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.config.max_new_tokens = 512
    data_collector = prepareTariningCollector()
    output_dir = global_file_path + global_models_folder + "/" + global_model_name

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        optim=optimizer,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="/out/",
        logging_steps=100,
        evaluation_strategy='steps',
        learning_rate=2e-5,
        save_steps=1000,
        predict_with_generate=True,
        eval_steps=100,
        load_best_model_at_end=True,
        save_total_limit=5,
        per_device_train_batch_size=Batch_Size,
        per_device_eval_batch_size=Batch_Size,
        num_train_epochs=Num_Training_Epochs,
        fp16=True,
        dataloader_num_workers=4,
        seed=30
    )

    log_manager.setTrainingArguments(training_arguments)

    # log_all_finetunning_params(training_arguments)

    train_tokenized, test_tokenized, eval_tokenized = datasetInfo(dataset_language, enhanced_dataset)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collector,
        compute_metrics=metric_eval.compute_metricBlue
    )

    showModelInfo(model)
    log_manager.write_logs("\n\n[Finetunning Started] - MODEL: ")
    log_manager.write_logs(model_checkpoint)
    log_manager.write_logs("\n\n")
    log_manager.write_training_Args()

    setTokenizerParalelism("true")

    trainer.train()
    print("Fine tunning has finished!")
    # germna_T5_textToGloss;
    save_model_path = log_manager.get_model_full_path() + "/finalModel"
    trainer.save_model(save_model_path)
    print("Model will be saved in ", save_model_path)
    # sper sa nu aiba iar loss 0


def trainModel_LoRA():
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, torch_dtype=torch.bfloat16)
    print("Training Batch size: ")
    Batch_Size = int(input())
    print("\nTraining epochs: ")
    Num_Training_Epochs = int(input())
    dataset_language = input("Choose dataset (german/american): ")
    data_collector = prepareTariningCollector()

    lora_config = LoraConfig(
        r=32,  # Note the rank (r) hyper-parameter, which defines the rank/dimension of the adapter to be trained.
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.35,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    peft_model = get_peft_model(original_model,
                                lora_config)
    showModelInfo(peft_model)

    train_tokenized, test_tokenized, eval_tokenized = datasetInfo(dataset_language,True)
    output_dir = global_file_path + global_models_folder + "/" + global_model_name

    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        #auto_find_batch_size=True,
        learning_rate=1e-3,
        logging_steps=500,
        eval_steps=1000,
        save_steps=2000,
        evaluation_strategy='steps',
        save_total_limit=5,
        per_device_train_batch_size=Batch_Size,
        #per_device_eval_batch_size=Batch_Size,
        num_train_epochs=Num_Training_Epochs,
        fp16=True,
        dataloader_num_workers=4,
        seed=30,
    )

    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        #data_collator=data_collector,
        #compute_metrics=metric_eval.compute_metricBlue
    )

    setTokenizerParalelism("false")
    log_manager.write_logs("\n\n[Lora - Finetunning Started] - MODEL: ")
    log_manager.write_logs(model_checkpoint)
    log_manager.write_logs("\n\n")

    start = input("Start Training (Y/N)?")

    if start == "Y" or start == "y":
        peft_trainer.train()
        print("Finetunning has finished!")
        peft_model_path = log_manager.get_model_full_path() + "/finalLoraModel"
        print("Saving path for lora :", peft_model_path)
        peft_trainer.model.save_pretrained(peft_model_path)
        tokenizer.save_pretrained(peft_model_path)

def trainModel_NEFT():
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    model.config.max_new_tokens = 500
    data_collector = prepareTariningCollector()

    print("Training Batch size: ")
    Batch_Size = int(input())
    print("\nTraining epochs: ")
    Num_Training_Epochs = int(input())

    training_arguments = Seq2SeqTrainingArguments(
        output_dir="/out/Models_23_02_24/GermanXLM_NEFT_Blue_v1",
        warmup_steps=200,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-5,
        evaluation_strategy='steps',
        logging_dir="/out/",
        save_steps=1000,
        predict_with_generate=True,
        eval_steps=200,
        per_device_train_batch_size=Batch_Size,
        per_device_eval_batch_size=Batch_Size,
        num_train_epochs=Num_Training_Epochs,
        neftune_noise_alpha=5,
        fp16=True,
        dataloader_num_workers=4,
        seed=30,
    )

    # log_all_finetunning_params(training_arguments)

    train_dataset, test_dataset = getDatasets()

    peft_params = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    write_logs_to_file("Lora config done.",
                       file_path="/out/Models_27_02_24/errors.txt")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collector,
        formatting_func=formatting_function,
        compute_metrics=metric_eval.compute_metricBlue,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        peft_config=peft_params,
    )
    write_logs_to_file("SFTTRainer done.",
                       file_path="/out/Models_27_02_24/errors.txt")
    trainer.train()

    save_model_path = metrics_global_file_path + "/finalModel"
    trainer.save_model(save_model_path)


def inference_LoRAModel():
    peft_model_path = log_manager.get_model_full_path() + "/finalLoraModel"
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, torch_dtype=torch.bfloat16)
    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(peft_model_path,torch_dtype=torch.bfloat16)

    peft_model = PeftModel.from_pretrained(peft_model_base, peft_model_path, torch_dtype=torch.bfloat16, is_trainable=False)
    peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

    showModelInfo(peft_model)
    showModelInfo(original_model)

    textInput = input("Text: ")
    while textInput != 'q':
        prompt = f"""
           Translate this German text into his gloss form: 
           {textInput}"""

        input_ids = peft_tokenizer(prompt, return_tensors="pt").input_ids

        original_model_outputs = original_model.generate(input_ids=input_ids,
                                                         generation_config=GenerationConfig(max_new_tokens=200,
                                                                                            num_beams=1))
        original_model_text_output = peft_tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

        peft_model_outputs = peft_model.generate(input_ids=input_ids,
                                                 generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
        peft_model_text_output = peft_tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

        print(f'\nORIGINAL MODEL: {original_model_text_output}')
        print(f'\nPEFT MODEL: {peft_model_text_output}')

        textInput = input("Text: ")

def inference():
    model_name = input("Choose model name: ")
    print("Text:")
    text = input()

    generate_kwargs = {"do_sample": True, "temperature": 0.0001, "max_new_tokens": max_target_length}

    model = pipeline("translation",
                     model=log_manager.get_model_full_path() + "/" + model_name,
                     # max_new_tokens=500,
                     max_length=512
                     )
    while text != 'q':
        translate = prefix + text
        glosses = model(translate)
        print(glosses)
        text = input("Text: ")


def showModelInfo(model):
    print("\n\nModel Param Information \n\n")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")


def compareModels():
    modelGerman = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    showModelInfo(modelGerman)

    modelTelekom = AutoModelForSeq2SeqLM.from_pretrained("GermanT5/t5-efficient-gc4-all-german-small-el32")
    showModelInfo(modelTelekom)

def metricEvaluator():

    model_name = input("Model name: ")
    model_type = input("Model Type(lora/ normal):")
    fullPath = log_manager.get_model_full_path() + "/" + model_name
    print("Calculating metric for ", fullPath)
    dataset_percent = float(input("How much to use from dataset(0 - 1.0): "))
    log_step = int(input("Log steps: "))
    dataset_language = input("Dataset Language (german/american): ")

    if "lora" in model_type:
        peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(fullPath, torch_dtype=torch.bfloat16)

        model = PeftModel.from_pretrained(peft_model_base, fullPath, torch_dtype=torch.bfloat16, is_trainable=False)
        eval_tokenizer = AutoTokenizer.from_pretrained(fullPath)

    elif "normal" in model_type:
        model = AutoModelForSeq2SeqLM.from_pretrained(fullPath, torch_dtype=torch.bfloat16)
        eval_tokenizer = AutoTokenizer.from_pretrained(fullPath)

    train_dataset, test_dataset, eval_dataset = datasetInfo(dataset_language)

    metric_eval.calculate_score(model, eval_tokenizer, test_dataset, dataset_percent, log_step)


def main_function():
    print("Zaha Trainer options\n")
    print(
        "1 - TrainModel_Seq2Seq\n2 - TrainModel_LoRA\n3 - Inference finnetuned model\n3 - Compare Models \n4 - Dataset Info\n5 - Log finetunning params\n"
        "6 = Log training arguments\n7 - Inference Lora Model\n8 - Metric Calculator",)

    option = int(input("Option: "))

    # match option:
    #     case 1:
    #         trainModel_Seq2SeqTrainer()
    #     case 2:
    #         global model_checkpoint
    #         model_checkpoint = decoder_model_checkpoint
    #         trainModel_NEFT()
    #     case 3:
    #         inference()
    #     case 4:
    #         compareModels()
    #     case 5:
    #         datasetInfo()
    #     case 6:
    #         log_manager.write_training_Args()
    #     case 7:
    #         trainModel_LoRA()
    if option == 1:
        trainModel_Seq2SeqTrainer()
    elif option == 2:
        trainModel_LoRA()
    elif option == 3:
        inference()
    elif option == 4:
        compareModels()
    elif option == 5:
        datasetInfo()
    elif option == 6:
        log_manager.write_training_Args()
    elif option == 7:
        inference_LoRAModel()
    elif option == 8:
        # tokenized_train, tokenized_test = datasetInfo("american")
        # print(tokenized_train)
        # print(tokenized_test)
        #return tokenized_train_dataset, tokenized_test_dataset
        metricEvaluator()

    else:
        print("Invalid option selected")

main_function()
