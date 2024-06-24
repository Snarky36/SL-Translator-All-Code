import base64

from BackEndApp.DataTransferObjects.TranslationObject import TranslationObject
from flask import request, jsonify, Blueprint, send_file
import random

from GlossToPoseModel.GlossVisualizing import computePoseGif
from Seq2SeqFinetunning.ModelManager import ModelManager

translationController = Blueprint('translationController', __name__)

model = ModelManager()

@translationController.route('/translate', methods=['GET'])
def translate():
    german_text = request.args.get('sentence')
    model_type = request.args.get('model-type')

    if model_type is None or model_type == '':
        return jsonify({'error': 'No model_type provided'}), 400
    if not german_text:
        return jsonify({'error': 'No text provided'}), 400

    model.set_model(model_type)

    gloss_sentence = model.inference(german_text)

    try:
        gif_path = computePoseGif(gloss_sentence, 5, True)


        with open(gif_path, 'rb') as gif_file:
            translation_video_data = gif_file.read()
            translation_video_data_base64 = base64.b64encode(translation_video_data).decode('utf-8')

        translation_object = TranslationObject(
            translated_text=gloss_sentence,
            translation_video=translation_video_data_base64
        )
        return jsonify(translation_object.to_dict()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500