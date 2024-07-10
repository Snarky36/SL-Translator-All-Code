class TranslationObject:
    def __init__(self, translated_text,
                 translation_video,
                 total_time,
                 gloss_time,
                 searching_time,
                 concatenation_time):
        self.translated_text = translated_text
        self.translation_video = translation_video
        self.total_response_time = total_time
        self.gloss_response_time = gloss_time
        self.searching_response_time = searching_time
        self.concatenation_response_time = concatenation_time
    def to_dict(self):
        return {
            'translated_text': self.translated_text,
            'translation_video': self.translation_video,
            'total_response_time': self.total_response_time,
            'gloss_response_time': self.gloss_response_time,
            'searching_response_time': self.searching_response_time,
            'concatenation_response_time': self.concatenation_response_time,
        }