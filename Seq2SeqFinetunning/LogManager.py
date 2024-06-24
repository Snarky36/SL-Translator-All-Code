import os
import numpy as np
from transformers import (TrainingArguments)

class LogManager():
    def __init__(self, path, modelsFolder="/Models", modelName="", trainingArguments=TrainingArguments(output_dir="/out/"), modelType=""):
        self.trainingArguments = trainingArguments
        self.path = path
        self.modelName = modelName
        self.modelsFolder = modelsFolder
        self.modelPath = self.path + modelsFolder + "/" + modelName
        self.metricsPath = self.path + modelsFolder + "/metrics.txt"
        self.errorPath = self.path + "errors.txt"
        self.logsPath = self.path + self.modelsFolder + "/logs.txt"
        self.logsTrainingPath = self.path + modelsFolder + "/trainingLogs.txt"
        self.inferencePath = self.path + modelsFolder + "/" + modelName + "/inference.txt";
        self.metricsPathTitle = ""
        self.logsTitle = ""
        self.modelType = modelType


    def check_and_createPath(self):
        if not os.path.exists(self.path):
            print("Creating Path:", self.path)
            os.mkdir(self.path)
        if not os.path.exists(self.modelPath):
            print("Creating modelPath:", self.modelPath)
            os.mkdir(self.modelPath)

    def write_metrics(self, score):
        if self.metricsPathTitle == "":
            self.metricsPathTitle = "Scores: " + str(self.trainingArguments.optim) + "\n"
            self.write_logs_to_file(self.metricsPathTitle, self.metricsPath)
        self.write_logs_to_file(score, self.metricsPath)

    def write_errors(self, errors):
        self.write_logs_to_file(errors, self.errorPath)

    def write_logs(self, logs):
        if self.logsTitle == "":
            self.logsTitle = "Logs for model :" + self.modelName + "\n\n"
            self.write_logs_to_file(self.logsTitle, self.logsPath)
        self.write_logs_to_file(logs, self.logsPath)

    def write_training_logs(self, logs):
        self.write_logs_to_file(logs, self.logsTrainingPath)
        
    def write_training_Args(self):
        self.write_logs("\n\n======================\n Finetune Process Params \n======================\n")
        self.write_logs("ModelType = " + self.modelType + "\n")
        self.write_logs("Model Name = " + self.modelName + "\n")
        self.write_logs("neftune_noise_alpha=" + str(self.trainingArguments.neftune_noise_alpha) + "\n")
        self.write_logs("output_dir=" + self.trainingArguments.output_dir + "\n")
        self.write_logs("warmup_steps=" + str(self.trainingArguments.warmup_steps) + "\n")
        self.write_logs("warmup_ratio=" + str(self.trainingArguments.warmup_ratio) + "\n")
        self.write_logs("weight_decay=" + str(self.trainingArguments.weight_decay) + "\n")
        self.write_logs("evaluation_strategy=" + self.trainingArguments.evaluation_strategy + "\n")
        self.write_logs("learning_rate=" + str(self.trainingArguments.learning_rate) + "\n")
        self.write_logs("predict_with_generate=" + str(self.trainingArguments.predict_with_generate) + "\n")
        self.write_logs("per_device_train_batch_size=" + str(self.trainingArguments.per_device_train_batch_size) + "\n")
        self.write_logs("per_device_eval_batch_size=" + str(self.trainingArguments.per_device_eval_batch_size) + "\n")
        self.write_logs("num_train_epochs=" + str(self.trainingArguments.num_train_epochs) + "\n")
        self.write_logs("dataloader_num_workers=" + str(self.trainingArguments.dataloader_num_workers) + "\n")
        self.write_logs("seed=" + str(self.trainingArguments.seed) + "\n")
        self.write_logs("========================\n")

    def setTrainingArguments(self, trainingArguments):
        self.trainingArguments = trainingArguments
    def convert_and_write(self, logs):
        new_logs = ""
        for s in logs:
            t = s.decode('iso-8859-1')
            new_logs += t.encode('utf-8')
        return new_logs

    def write_logs_to_file(self, logs, file_path='/out/logs.txt'):
        with open(file_path, 'a') as file:
            try:
                file.write(str(logs))
            except NameError:
                print("Could not write some german characters:\n")
                print(NameError)
                file.write(str(self.convert_and_write(logs)))

    def get_model_full_path(self):
        return self.modelPath

    def write_inference_to_file(self, results, references):
        self.write_logs_to_file("Predictions                               References", self.inferencePath)
        for index in range(0, len(results)):
            log = "\n"+"".join(results[index]) + "    ||||    " + "".join(references[index])
            self.write_logs_to_file(log, self.inferencePath)


    def set_inferencePath(self, model_version):
        self.inferencePath = self.path + self.modelsFolder + "/" + self.modelName + "/inference_" + model_version + ".txt"

    def get_inferencePath(self):
        return self.inferencePath

    def setModelType(self, checkpoint):
        self.modelType = checkpoint