import csv
import os
from collections import defaultdict
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, correct_wrists, pose_normalization_info
import numpy as np
from scipy.spatial.distance import cdist
from pose_format.numpy import NumPyPoseBody

poses_dir_path = os.path.join(os.path.dirname(__file__), f"""assets/signsuisse""")
csv_name = "index.csv"
missing_glosses = set()

def read_csv_data(folder_path, csv_name):
    with open(os.path.join(folder_path, csv_name), mode='r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    return rows


def create_gloss_to_pose_dictionary(rows):
    gloss_dictionary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        # print("row", row)
        gloss = row["glosses"].lower()
        gloss_dictionary[row['spoken_language']][row['signed_language']][gloss].append({
            "path": row['path'],
            "start": row['start'],
            "end": row['end']
        })

    return gloss_dictionary


def find_pose(gloss_dictionary, gloss: str, spoken_language="de", signed_language="sgg"):
    lower_gloss = gloss.lower()

    if lower_gloss in gloss_dictionary[spoken_language][signed_language]:
        row = gloss_dictionary[spoken_language][signed_language][lower_gloss][0]
        print("gloss:", gloss, row)
        pose_path = poses_dir_path + "/" + row["path"]
        with open(pose_path, "rb") as f:
            return Pose.read(f.read())
    else:
        missing_glosses.add(gloss)
    return ""


def create_sequence_of_poses(glosses, gloss_dictionary, spoken_language="de", signed_language="sgg"):
    pose_sentence = []
    for gloss in glosses:
        pose = find_pose(gloss_dictionary, gloss, spoken_language, signed_language)
        if pose != "":
            pose_sentence.append(pose)
    if len(pose_sentence) == 0:
        print("Can't find any pose for the sentence: ", glosses)

    return pose_sentence


def create_signLanguage_sentence(pose_sentence):
    poses = [reduce_holistic(pose) for pose in pose_sentence]

    # se normalizeaza in functie de anumite puncte. Umeri sau mean si standard deviation pentru fiecare punct sau toate deodata
    type_of_normalization = ["shoulders_key_points", "all_key_points", "each_key_point"]
    poses = [normalize_pose(pose, type_of_normalization[0]) for pose in poses]

    # Excludere frameuri in care confidenta mainilor e prea mica si nu putem decide clar daca ele apar sau nu.
    print('Trimming poses...')
    poses = [trim_pose(p, i > 0, i < len(poses) - 1) for i, p in enumerate(poses)]

    # Concatenate all poses
    print('Smooth concatenating poses...')
    final_pose = smooth_concatenate_poses(poses)

    # Correct the wrists (should be after smoothing)
    print('Correcting wrists...')
    final_pose = correct_wrists(final_pose)

    print('Scaling pose...')
    new_width = 500
    shift = 1.25
    shift_vec = np.full(shape=(final_pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
    final_pose.body.data = (final_pose.body.data + shift_vec) * new_width
    final_pose.header.dimensions.height = final_pose.header.dimensions.width = int(new_width * shift * 2)

    return final_pose

def normalize_pose(pose: Pose, type_of_normalization="shoulders_key_points"):
    if type_of_normalization == "all_key_points":
        mean, standard_deviation = pose.normalize_distribution()
        print("Normalized the pose\n mean:", mean, "\n standard_deviation:", standard_deviation)

        return pose
    elif type_of_normalization == "shoulders_key_points":
        normalized_pose = pose.normalize(pose_normalization_info(pose.header))

        return normalized_pose
    elif type_of_normalization == "each_key_point":
        mean, standard_deviation = pose.normalize_distribution(axis=(0, 1, 2))
        print("Normalized the pose\n mean:", mean, "\n standard_deviation:", standard_deviation)

        return pose


def trim_pose(pose, start=True, end=True,
              min_confidence=0):  # min_confidence e cat de sigur vrea sa fiu cand aleg un frame care contine maini.
    if len(pose.body.data) == 0:
        print("data for this pose is empty! => ")
        return pose

    wrist_indexes = [
        pose.header._get_point_index('LEFT_HAND_LANDMARKS', 'WRIST'),
        pose.header._get_point_index('RIGHT_HAND_LANDMARKS', 'WRIST')
    ]
    either_hand = pose.body.confidence[:, 0, wrist_indexes].sum(axis=1) > min_confidence
    print("either_hand", either_hand)
    first_non_zero_index = np.argmax(either_hand) if start else 0
    last_non_zero_index = (len(either_hand) - np.argmax(either_hand[::-1]) - 1) if end else len(either_hand)

    pose.body.data = pose.body.data[first_non_zero_index:last_non_zero_index]
    pose.body.confidence = pose.body.confidence[first_non_zero_index:last_non_zero_index]
    return pose


def smooth_concatenate_poses(poses, padding=0.20) -> Pose:
    if len(poses) == 0:
        print("There are no poses")
        return None

    start = 0
    for i, pose in enumerate(poses):
        print('Processing', i + 1, 'of', len(poses), '...')
        if i != len(poses) - 1:
            pose1_end, pose2_start = get_best_point_connection(poses[i], poses[i + 1])
        else:
            pose1_end = len(pose.body.data)
            pose2_start = None

        pose.body = pose.body[start:pose1_end]
        start = pose2_start

    padding_pose = pad_pose_frames(padding, poses[0])

    final_pose = create_final_pose(poses, padding_pose)

    return final_pose


def get_best_point_connection(pose1, pose2, kernel_size_percent=0.3, metric="euclidean"):

    p1_size = int(len(pose1.body.data) * kernel_size_percent)
    p2_size = int(len(pose2.body.data) * kernel_size_percent)

    #motivul pentru care ales ultimele miscari din primul cadrul si primele miscari din al doilea e ca sa compar
    #cat de diferite sunt pentru a face o interpolare si a adauga cadre ca sa iasa video mai smooth
    #ultimele miscari dintr-un cadru si primele din celalalt ar trebui sa fie apropiate deaorece reprezinta o miscare continua
    # daca ele sunt diferite tare incerc sa fac trecerea mai smooth sa nu se vada sacadat
    pose1_last_movements = pose1.body.data[len(pose1.body.data) - p1_size:]
    pose2_first_movements = pose2.body.data[:p2_size]

    pose1_last_vectors = pose1_last_movements.reshape(len(pose1_last_movements), -1)
    pose2_first_vectors = pose2_first_movements.reshape(len(pose2_first_movements), -1)

    distances_matrix = cdist(pose1_last_vectors, pose2_first_vectors, metric=metric)
    #aleg linia si coloana unde se afla cel mai mic element din matricea de distante si linia va corespunde cu
    # indexul elementului din last_vector si coloana cu indexul din pose1_first_vector
    min_index = np.unravel_index(np.argmin(distances_matrix, axis=None), distances_matrix.shape) # min_index = (i,j)
    last_index = len(pose1.body.data) - p1_size + min_index[0]

    #calculal last_index care o sa corespunda cu pozitia celui mai bun punct unde putem adauga date in pose1.body
    # la fel facem si cu pozitia din pose2 adica min_index[1]
    return last_index, min_index[1]


def pad_pose_frames(time: float, pose) -> NumPyPoseBody:
    #iau cate un pose si il adaug un numar aditional de frameuri cu un procent de Time*initial fps
    #practic il mai prelunges cu n% frameuri per secunda
    fps = pose.body.fps
    padding_frames = int(time * fps)
    data_shape = pose.body.data.shape
    return NumPyPoseBody(fps=fps, data=np.zeros(shape=(padding_frames, data_shape[1], data_shape[2], data_shape[3])),
                         confidence=np.zeros(shape=(padding_frames, data_shape[1], data_shape[2])))

def create_final_pose(poses, padding, interpolation='linear') -> Pose:
    for (index, pose) in enumerate(poses):
        if index < len(poses):
            pose.body.data = np.concatenate((pose.body.data, padding.data))
            pose.body.confidence = np.concatenate((pose.body.confidence, padding.confidence))

    final_pose_data = np.concatenate([pose.body.data for pose in poses])
    final_pose_confidence = np.concatenate([pose.body.confidence for pose in poses])
    final_pose_body = NumPyPoseBody(fps=poses[0].body.fps, data=final_pose_data, confidence=final_pose_confidence)
    final_pose_body = final_pose_body.interpolate(kind=interpolation)

    return Pose(header=poses[0].header, body=final_pose_body)


def get_glossDict():
    rows = read_csv_data(poses_dir_path, csv_name)
    gloss_dictionary = create_gloss_to_pose_dictionary(rows)
    return gloss_dictionary

def get_word_to_gloss_dict():
    rows = read_csv_data(poses_dir_path, csv_name)
    gloss_to_word_dict = {}
    for row in rows:
        # print("row", row)
        gloss = row["glosses"].lower()
        word = row["words"].lower()
        gloss_to_word_dict[word] = gloss

    return gloss_to_word_dict

def PoseLooker(glosses, title):

    gloss_dictionary = get_glossDict()
    # print("gloss_disct", gloss_dictionary)
    pose_sentence = create_sequence_of_poses(glosses, gloss_dictionary)

    print("poses", pose_sentence)
    final_pose = create_signLanguage_sentence(pose_sentence)
    print("final_pose", final_pose)
    pose_path = os.path.join(os.path.dirname(__file__), f"""assets/GeneratedPoses/{title}.pose""")
    with open(pose_path, "wb") as f:
        final_pose.write(f)

    return missing_glosses

#PoseLooker(["Gut", "Morgen"], "Guten_Morgen__noTrim_shoulderNormalization.pose")
