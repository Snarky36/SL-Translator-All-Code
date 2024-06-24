import os

from transformers import pipeline

max_input_length = 256
max_target_length = 256
prefix = "Translate this German text into his gloss form: "
# model_name = input("Choose model name: ")
#
#
# generate_kwargs = {"do_sample": True, "temperature": 0.0001, "max_new_tokens": max_target_length}
#
# model = pipeline("translation",
#                  model="./Models/" + model_name,
#                  max_length=512
#                 )
#
# def inference(text):
#
#     translate = prefix + text
#     glosses = model(translate)
#
#     return glosses[0]["translation_text"]


class ModelManager():
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "Models/finalModelDe/")
        self.model = pipeline("translation",
                 model=model_path,
                 max_length=512
                )
        self.model_type = "T5"

    def set_model(self, model_name):
        if model_name == "normal_lstm":
            pass
        elif model_name == "normal_gru":
            pass
        elif model_name == "attention_lstm":
            pass
        elif model_name == "MT5_German":
            model_path = os.path.join(os.path.dirname(__file__), "Models/finalModelDe/")
            self.model = pipeline("translation",
                                  model=model_path,
                                  max_length=512
                                  )
            self.model_type = "T5"
        elif model_name == "MT5_German_Enhanched":
            pass
        elif model_name == "T5_Telekom_German":
            pass
        elif model_name == "MT5_English":
            pass
        elif model_name == "MT5_English_LoRA":
            pass

    def inference(self, text):
        if self.model_type == "T5":
            translate = prefix + text
            glosses = self.model(translate)

            return glosses[0]["translation_text"]

        return None