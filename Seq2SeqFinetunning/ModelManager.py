import gc
import os

import torch
from peft import PeftModel
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from SequenceToSequenceModel.ModelInference import set_seq_to_seq_model, inference_seq_to_seq
max_input_length = 256
max_target_length = 256
prefix = "Translate this German text into his gloss form: "

class ModelManager():
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "Models/")
        self.model = None
        self.model_type = "Transformer"
        self.peft_model = None
        self.peft_tokenizer = None

        self.tokenizer = None
        self.use_attention = None
        self.use_gru = None

    def set_model(self, model_name):
        gc.collect()
        torch.cuda.empty_cache()
        if model_name == "normal_lstm":

            self.model, self.tokenizer, self.use_attention, self.use_gru = set_seq_to_seq_model(path=self.model_path,
                                                                                                model_name="zaha-german-gloss-25-06-LSTM-V1-final-1.683",
                                                                                                respect_dataset_max_length=True,
                                                                                                use_attention=False,
                                                                                                use_gru=False,
                                                                                                trained_multiple_gpu=False,
                                                                                                hidden_size=1024,
                                                                                                num_layers=5
                                                                                                )
            self.model_type = "SeqToSeq"

        elif model_name == "normal_gru":
            self.model, self.tokenizer, self.use_attention, self.use_gru = set_seq_to_seq_model(path=self.model_path,
                                                                                                model_name="zaha-german-gloss-25-06-GRU-V1-final-2.060",
                                                                                                respect_dataset_max_length=True,
                                                                                                use_attention=False,
                                                                                                use_gru=True,
                                                                                                trained_multiple_gpu=False,
                                                                                                hidden_size=1024,
                                                                                                num_layers=5
                                                                                                )
            self.model_type = "SeqToSeq"
        elif model_name == "attention_lstm":
            self.model, self.tokenizer, self.use_attention, self.use_gru = set_seq_to_seq_model(path=self.model_path,
                                                                                                model_name="zaha-german-gloss-21-06-LSTM-Attention_LowModel_V1-0.533",
                                                                                                respect_dataset_max_length=True,
                                                                                                use_attention=True,
                                                                                                use_gru=False,
                                                                                                trained_multiple_gpu=True,
                                                                                                hidden_size=1024,
                                                                                                num_layers=4
                                                                                                )
            self.model_type = "SeqToSeq"
        elif model_name == "T5_Telekom_German":
            model_path = os.path.join(os.path.dirname(__file__), "Models/Deutsche-Telekom-T5/")
            self.model = pipeline("translation",
                                  model=model_path,
                                  max_length=512,
                                  device="cuda"
                                  )
            self.model_type = "Transformer"
        elif model_name =="T5_Telekom_German_Enhanced":
            model_path = os.path.join(os.path.dirname(__file__), "Models/T5_GermanModel_EnhancedV3_turbo/")
            self.model = pipeline("translation",
                                  model=model_path,
                                  max_length=512,
                                  device="cuda"
                                  )
            self.model_type = "Transformer"
        elif model_name == "MT5_German_Enhanched":
            model_path = os.path.join(os.path.dirname(__file__), "Models/T5_GermanMT5Model_Enhanced_turbo/")
            self.model = pipeline("translation",
                                  model=model_path,
                                  max_length=512,
                                  device="cuda"
                                  )
            self.model_type = "Transformer"
        elif model_name == "MT5_German_LoRA":
            base_model_path = os.path.join(os.path.dirname(__file__), "Models/T5_GermanModel_EnhancedV3_turbo")
            peft_model_path = os.path.join(os.path.dirname(__file__), "Models/MT5_German_LoRA")

            peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(peft_model_path, torch_dtype=torch.bfloat16)

            self.peft_model = PeftModel.from_pretrained(peft_model_base, peft_model_path, torch_dtype=torch.bfloat16,
                                                        is_trainable=False)
            self.peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

            self.model_type = "Transformer_LoRA"
        elif model_name == "MT5_English":
            model_path = os.path.join(os.path.dirname(__file__), "Models/T5_EnglishModel/")
            self.model = pipeline("translation",
                                  model=model_path,
                                  max_length=512,
                                  device="cuda"
                                  )
            self.model_type = "Transformer"

        elif model_name == "MT5_English_LoRA":
            peft_model_path = os.path.join(os.path.dirname(__file__), "Models/MT5_English_LoRA")

            peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(peft_model_path, torch_dtype=torch.bfloat16)

            self.peft_model = PeftModel.from_pretrained(peft_model_base, peft_model_path, torch_dtype=torch.bfloat16,
                                                        is_trainable=False)
            self.peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

            self.model_type = "Transformer_LoRA"




    def inference(self, text):
        if self.model_type == "Transformer":
            translate = prefix + text
            glosses = self.model(translate)

            return glosses[0]["translation_text"]

        if self.model_type == "Transformer_LoRA":
            prompt = f"""
                       Translate this German text into his gloss form: 
                       {text}"""

            input_ids = self.peft_tokenizer(prompt, return_tensors="pt").input_ids

            peft_model_outputs = self.peft_model.generate(input_ids=input_ids,
                                                     generation_config=GenerationConfig(max_new_tokens=200,
                                                                                        num_beams=1))
            peft_model_text_output = self.peft_tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

            print(f'\nPEFT MODEL: {peft_model_text_output}')
            return peft_model_text_output

        if self.model_type == "SeqToSeq":
            glosses = inference_seq_to_seq(model=self.model,
                                           tokenizer=self.tokenizer,
                                           use_attention=self.use_attention,
                                           use_gru=self.use_gru,
                                           apply_softmax=True,
                                           text=text
                                           )
            print(f'\nSeqToSeqModel: {glosses}')
            return glosses

        return None