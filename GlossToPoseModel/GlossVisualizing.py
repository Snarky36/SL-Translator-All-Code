import os

from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from GlossToPoseModel.PoseLooker import PoseLooker, get_glossDict, get_word_to_gloss_dict
import Levenshtein
from sentence_transformers import SentenceTransformer, util
import numpy as np


speed_up = True

#gloss_sentence = "SUED BAYERN IX WARNUNG DEUTSCH WETTER DIENST HEUTE NACHT SUED AUCH GEWITTER"

def compute_title(gloss_sentence):
    gloss_sentence = gloss_sentence
    title = ""
    for gloss in gloss_sentence.split(" ")[:3]:
        title += gloss
        title += "_"
    title += "_trim_shoulderNorm"

    return title

def word2vec(word):
    from collections import Counter
    from math import sqrt
    cw = Counter(word)
    sw = set(cw)
    lw = sqrt(sum(c*c for c in cw.values()))

    return cw, sw, lw

def cosine_distance(v1, v2):
    common = v1[1].intersection(v2[1])
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]


def calculateDistance(word1, word2, typeOfDistance="cosine"):
    if typeOfDistance == "Levenshtein":
        return 1 / Levenshtein.distance(word1, word2)
    else:
        vector1 = word2vec(word1)
        vector2 = word2vec(word2)
        return cosine_distance(vector1, vector2)

def compute_word_confidence(word, gloss, cosine_model_score):
    vector1 = word2vec(word)
    vector2 = word2vec(gloss)
    cosine_dist = cosine_distance(vector1, vector2)
    levenshtein_dist = calculateDistance(word,gloss,"Levenshtein")

    return 0.1 * levenshtein_dist + 0.4 * cosine_dist + 0.5 * cosine_model_score



def preprocess_sentence(gloss_sentence, k=5, spoken_language="de", sing_language="sgg", model_discriminator="all-MiniLM-L6-v2"):
    glosses = [gloss for gloss in gloss_sentence.split(" ")]
    new_gloss_sentence = []
    skipped_glosses = []
    model = SentenceTransformer(model_discriminator)

    all_dict = get_glossDict()
    word_to_gloss_dict = get_word_to_gloss_dict()
    for gloss in glosses:
        gloss = gloss.lower()
        if gloss not in all_dict[spoken_language][sing_language]:
            if gloss in word_to_gloss_dict.keys():
                new_gloss_sentence.append(word_to_gloss_dict[gloss])
                continue

            print("Missing Gloss", gloss)

            top_k_words = []
            artificial_added_words = 0
            for word in all_dict[spoken_language][sing_language]:
                if gloss in word:
                    top_k_words.append((word, 1.0))
                    artificial_added_words += 1
                score = calculateDistance(gloss, word) #aici fac distanta cosinus
                top_k_words.append((word, score))

            top_k_words.sort(key=lambda x: x[1], reverse=True)
            k_words = artificial_added_words + k
            print(top_k_words[:k_words])
            top_k_words = [word for word, _ in top_k_words[:k_words]]

            scores = []
            gloss_embedding = model.encode(gloss, convert_to_tensor=True)
            for word in top_k_words:
                embedding = model.encode(word, convert_to_tensor=True)
                cosine_model_score = util.cos_sim(gloss_embedding, embedding)[0][0].item()
                confidence = compute_word_confidence(word, gloss, cosine_model_score)
                scores.append(confidence)
                print(word, ":", confidence)
            max_index = 0
            if len(scores) != 0:
                max_index = np.argmax(scores)
            word_with_highest_score = top_k_words[max_index]
            print("bestWord:", word_with_highest_score)
            if(scores[max_index] > 0.5):
                new_gloss_sentence.append(word_with_highest_score)
            else:
                skipped_glosses.append(gloss)
        else:
            new_gloss_sentence.append(gloss)

    return new_gloss_sentence, skipped_glosses


def computePoseGif(gloss_sentence, words_to_process_while_looking, speed_up=True):

    title = compute_title(gloss_sentence)
    print("Title", title)
    glosses, skipped_glosses = preprocess_sentence(gloss_sentence=gloss_sentence, k=words_to_process_while_looking)
    print(glosses)
    missing_glosses = PoseLooker(glosses, title)

    pose_path = os.path.join(os.path.dirname(__file__), f"""assets/GeneratedPoses/{title}.pose""")
    gif_path = os.path.join(os.path.dirname(__file__), f"""assets/GenereatedGifs/{title}.gif""")

    with open(pose_path, "rb") as f:
        p = Pose.read(f.read())

    #Dau un resize sa mearga mai repede treaba
    if speed_up:
        scale = p.header.dimensions.width / 256
        p.header.dimensions.width = int(p.header.dimensions.width / scale)
        p.header.dimensions.height = int(p.header.dimensions.height / scale)
        p.body.data = p.body.data / scale

    v = PoseVisualizer(p)

    v.save_gif(gif_path, v.draw())

    print("From this sentence this glosses could not be found or changed", skipped_glosses, missing_glosses)

    return gif_path

#computePoseGif(gloss_sentence,5)
