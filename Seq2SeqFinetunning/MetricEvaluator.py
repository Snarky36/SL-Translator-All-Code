from datasets import load_dataset, load_metric
import sacrebleu
import numpy as np
import evaluate
from transformers import GenerationConfig
from bert_score import BERTScorer
from transformers import BertTokenizer, BertForMaskedLM, BertModel
class MetricEvaluator():
    def __init__(self, tokenizer, log_manager):
        self.log_manager = log_manager
        self.tokenizer = tokenizer
        self.rouge = evaluate.load("rouge")
        self.sacrebleu = evaluate.load("sacrebleu")

    def postprocess_text(self, preds, labels):
        self.log_manager.write_training_logs("\n\n[FinetuneV2] Preds: ")
        self.log_manager.write_training_logs(preds)
        print("Preds: ", preds[1])
        print("\n\nLabels ", labels[1])
        self.log_manager.write_training_logs("\n\n[FinetuneV2] Labels: ")
        self.log_manager.write_training_logs(labels)

        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metricBlue(self, eval_pred):
        metric = load_metric("sacrebleu", trust_remote_code=True)
        preds, labels = eval_pred

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)

        # write_logs_to_file("Started Metrics for model = ", file_path='/out/metriclogs.txt')
        # write_logs_to_file(model_checkpoint, file_path='/out/metriclogs.txt')
        self.log_manager.write_metrics("\n\nBlue Score= ")
        self.log_manager.write_metrics(result["score"])

        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def compute_metricRouge(self, eval_pred):
        print("EVAL_PREDS: ", eval_pred)
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        print("\n\nDecoded_Labels = ", decoded_labels[1])
        print("\n\nPredictions = ", decoded_preds)
        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        print("\n\n Rouge score", result)

        self.log_manager.write_metrics("\nRouge scroe:")
        self.log_manager.write_metrics(str(result))

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def calculate_blue(self, references, predictions):
        blue_score = self.sacrebleu.compute(predictions=predictions, references=references)
        print("evals: ", list(blue_score.keys()))
        print("blue_Score = ", blue_score["score"])
        print("precision(1/2/3/4 grams) = ", blue_score["precisions"])
        print("all metrics = ", blue_score)

        return blue_score



    def calculate_rouge(self, references, predictions):
        rouge = self.rouge.compute(predictions=predictions, references=references)
        print("rouge scores", rouge)

        return rouge

    def calculate_bertScore(self, references, predictions):
        bert_score = BERTScorer(model_type='bert-base-uncased')
        Precision, Recall, F1 = bert_score.score(predictions, references)
        print(f"BERTScore Precision: {Precision.mean():.4f}, Recall: {Recall.mean():.4f}, F1: {F1.mean():.4f}")

        return Precision, Recall, F1

    def calculate_score(self, model, tokenizer, dataset, dataset_percent, log_step, score="all"):

        log_path = self.log_manager.get_model_full_path() + "/evaluation_logs.txt"
        score_path = self.log_manager.get_model_full_path() + "/evaluation_scores.txt"

        dataset_input_number = int(len(dataset) * dataset_percent)

        metric_dataset = dataset.shuffle(seed=42).select(range(dataset_input_number))
        predictions = []
        references = []
        step = 0
        for data in metric_dataset:
            step +=1
            greater_truth = [data["target"]]
            references.append(greater_truth)

            prompt = f"""
                       Translate this German text into his gloss form: 
                       {data["text"]}"""

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids


            model_outputs_tokens = model.generate(input_ids=input_ids,
                                                     generation_config=GenerationConfig(max_new_tokens=200,
                                                                                        num_beams=1))

            model_text_output = tokenizer.decode(model_outputs_tokens[0], skip_special_tokens=True)

            predictions.append(model_text_output)

            if step % log_step == 0:
                print("Logging\n")
                print("\nprompt",prompt)
                print("\ninput_ids", input_ids)
                print("\nmodel_outputs_tokens", model_outputs_tokens)
                print("\nmodel_text_output", model_text_output)
                print("\nGreater truth", greater_truth)

                self.log_manager.write_logs_to_file("Logging\n", log_path)
                self.log_manager.write_logs_to_file(f"\nprompt {prompt}", log_path)
                self.log_manager.write_logs_to_file(f"\ninput_ids {input_ids}", log_path)
                self.log_manager.write_logs_to_file(f"\nmodel_outputs_tokens {model_outputs_tokens}", log_path)
                self.log_manager.write_logs_to_file(f"\nmodel_text_output {model_text_output}", log_path)
                self.log_manager.write_logs_to_file(f"\nGreater truth {greater_truth}", log_path)


            print("Steps:", step, "/", dataset_input_number)
            self.log_manager.write_logs_to_file(f"\nSteps: {step}/{dataset_input_number}", log_path)

        blue_score = ""
        rouge_score = ""
        P ,R, F1 = "", "", ""
        if(score == "blue"):
            blue_score = self.calculate_blue(references, predictions)
        if(score == "rouge"):
            rouge_score = self.calculate_rouge(references, predictions)
        if(score == "all"):
            blue_score = self.calculate_blue(references, predictions)
            rouge_score = self.calculate_rouge(references, predictions)
            P,R,F1 = self.calculate_bertScore(references, predictions)

        self.log_manager.write_logs_to_file(f"\nBlue Score: {blue_score}", score_path)
        self.log_manager.write_logs_to_file(f"\nRouge Score: {rouge_score}", score_path)
        self.log_manager.write_logs_to_file(f"BERTScore: \nPrecision: {P.mean():.4f} \nRecall: {R.mean():.4f} \nF1: {F1.mean():.4f}", score_path)

        self.log_manager.write_inference_to_file(results=predictions, references=references)


