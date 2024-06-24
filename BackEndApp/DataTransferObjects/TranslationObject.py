class TranslationObject:
    def __init__(self, translated_text, translation_video):
        self.translated_text = translated_text
        self.translation_video = translation_video

    def to_dict(self):
        return {
            'translated_text': self.translated_text,
            'translation_video': self.translation_video
        }