import matplotlib.pyplot as plt
import re
#from FinetunningV2 import datasetInfo

globally_registered_errors = {}
def read_txt_file(file_path):
    data_array = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            columns = line.strip().split("    ||||    ")
            if len(columns) >= 2:
                prediction = columns[0].strip()
                reference = columns[1].strip()
                data_array.append({"prediction": prediction, "reference": reference})

    return data_array


def calculate_Type1_repetitiveAditionale_simplu(data_array):
    duplicate_counts = {}
    for data in data_array:
        sentence = data["prediction"]
        print("Scetence:", sentence)
        glosses = re.split(r'\s+|[,\.]', sentence)
        print("Glosses:", glosses)
        duplicate_count = 0
        for i in range(len(glosses) - 1):
            if glosses[i] == glosses[i + 1]:
                duplicate_count += 1
        print("Duplicates:", duplicate_count)
        if duplicate_count > 0:
            duplicate_counts[sentence] = duplicate_count
    print("Values", duplicate_counts)
    return duplicate_counts

def calculate_damaging_type(gloss, references, sub_type="additional"):
    if sub_type == "damaging":
        return gloss not in references
    return gloss in references
        
def calculate_Type1_errors(data_array, sub_type="additional", include_random_glosses_error=False, return_just_sentences=False):
    total_words = 1
    ngram_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    ngram_sentences_count = {1: 0, 2: 0, 3: 0, 4: 0}

    sentences_with_errors = []
    sentence_with_error_count = 0

    for data in data_array:
        sentence_has_error = False
        sentence = data["prediction"]
        glosses = re.split(r'\s+|[,\.]', sentence)
        reference_glosses = re.split(r'\s+|[,\.]', data["reference"])
        total_words += len(glosses)

        # Count 1-gram duplicates
        error_detected = 0
        for i in range(len(glosses) - 1):
            if (calculate_damaging_type(glosses[i], reference_glosses, sub_type)
                    and glosses[i] == glosses[i + 1]):
                error_detected = 1
                ngram_counts[1] += 1
        ngram_sentences_count[1] += error_detected

        if error_detected:
            sentence_has_error = True

        # Count 2-gram duplicates
        error_detected = 0
        i = 0
        while i < len(glosses) - 3:
            if (calculate_damaging_type(glosses[i], reference_glosses, sub_type)
                    and glosses[i] != glosses[i + 1]):
                if (i + 2 < len(glosses)
                        and glosses[i] == glosses[i + 2]
                        and glosses[i + 1] == glosses[i + 3]):
                    ngram_counts[2] += 1
                    error_detected = 1
                    i += 1
            i += 1
        ngram_sentences_count[2] += error_detected
        if error_detected:
            sentence_has_error = True

        # Count 3-gram duplicates
        error_detected = 0
        i = 0
        while i < len(glosses) - 5:
            if (calculate_damaging_type(glosses[i], reference_glosses, sub_type)
                    and glosses[i] != glosses[i + 1]
                    and glosses[i] != glosses[i + 2]
                    and glosses[i + 1] != glosses[i + 2]):
                if (i + 3 < len(glosses)
                        and glosses[i] == glosses[i + 3]
                        and glosses[i + 1] == glosses[i + 4]
                        and glosses[i + 2] == glosses[i + 5]):
                    ngram_counts[3] += 1
                    error_detected = 1
                    i += 2
            i += 1
        ngram_sentences_count[3] += error_detected
        if error_detected:
            sentence_has_error = True

        # Count 4-gram duplicates
        error_detected = 0
        for i in range(len(glosses) - 7):
            if (calculate_damaging_type(glosses[i], reference_glosses, sub_type)
                    and glosses[i] != glosses[i + 1]
                    and glosses[i] != glosses[i + 2]
                    and glosses[i] != glosses[i + 3]
                    and glosses[i + 1] != glosses[i + 2]
                    and glosses[i + 1] != glosses[i + 3]
                    and glosses[i + 2] != glosses[i + 3]):
                if (i + 4 < len(glosses)
                        and glosses[i] == glosses[i + 4]
                        and glosses[i + 1] == glosses[i + 5]
                        and glosses[i + 2] == glosses[i + 6]
                        and glosses[i + 3] == glosses[i + 7]):
                    ngram_counts[4] += 1
                    error_detected = 1
        ngram_sentences_count[4] += error_detected
        if error_detected:
            sentence_has_error = True
        if sentence_has_error:
            sentences_with_errors.append(sentence)

        if include_random_glosses_error:
            unique_word_count = {word: 0 for word in glosses}
            for i, gloss in enumerate(glosses):
                if gloss not in reference_glosses:
                    if ((i == 0 or glosses[i - 1] != gloss)
                            and (i == len(glosses) - 1
                                 or glosses[i + 1] != gloss)):
                        unique_word_count[gloss] += 1
            for gloss in unique_word_count.keys():
                if unique_word_count[gloss] > 1:
                    sentence_with_error_count += 1
                    sentences_with_errors.append(sentence)
                    break

    if return_just_sentences:
        return sentences_with_errors

    if include_random_glosses_error:
        return total_words, ngram_counts, ngram_sentences_count, sentence_with_error_count

    return total_words, ngram_counts, ngram_sentences_count

def calculate_Type1_repetitiveAditionale_avansat(data_array):

    type1_repetitive_sentences=[]
    total_number_of_sentences = len(data_array)
    total_words, ngram_counts, ngram_sentences_count = calculate_Type1_errors(data_array)

    percentages = [0, 0, 0, 0]
    labels = ['1-gram', '2-gram', '3-gram', '4-gram']
    for i in ngram_counts:
        percentages[i - 1] = ngram_counts[i] / total_words * 100

    plt.bar(labels, percentages, color='skyblue')
    plt.xlabel('N-gram Duplicates')
    plt.ylabel('Percentage %')
    plt.title('Percentage of N-gram Duplicates from total of '+str(total_words) + 'words')
    plt.ylim(0, 100)
    for i, v in enumerate(percentages):
        plt.text(i, v + 1, "count:" + str(ngram_counts[i + 1]), ha='center', va='bottom')
        plt.text(i, v + 6, str("{:.3f}".format(percentages[i])) + '%', ha='center', va='bottom')
    plt.show()

    sentence_percentages = [0, 0, 0, 0]
    labels = ['1-gram', '2-gram', '3-gram', '4-gram']
    for i in ngram_counts:
        sentence_percentages[i - 1] = ngram_sentences_count[i] / total_number_of_sentences * 100

    plt.bar(labels, sentence_percentages, color='skyblue')
    plt.xlabel('N-gram Duplicates')
    plt.ylabel('Percentage %')
    plt.title('Percentage of sentences with N-gram error from total of ' + str(total_number_of_sentences) + ' sentences')
    plt.ylim(0, 100)
    for i, v in enumerate(sentence_percentages):
        plt.text(i, v + 1, "count:"+str(ngram_sentences_count[i + 1]), ha='center', va='bottom')
        plt.text(i, v + 6, str("{:.3f}".format(sentence_percentages[i]))+'%', ha='center', va='bottom')
    plt.show()

    return


def calculate_Type1_repetitiveDaunatoare(data_array: object) -> object:
    sentence_with_error = 0
    total_number_of_sentences = len(data_array)
    total_words, ngram_counts, ngram_sentences_count, sentence_with_error = calculate_Type1_errors(data_array, "damaging", True)


    percentages =[0, 0, 0, 0, 0]
    labels = ['random', '1-gram', '2-gram', '3-gram', '4-gram']
    percentages[0] = sentence_with_error / total_number_of_sentences * 100

    for i in ngram_sentences_count:
        percentages[i] = ngram_sentences_count[i] / total_number_of_sentences * 100

    plt.bar(labels, percentages, color='green')
    plt.xlabel('Number of sentences with errors')
    plt.ylabel('Percentage %')
    plt.title('Number of destructive repetitive glosses from a total of '
              + str(total_number_of_sentences) + ' sentences')
    plt.ylim(0, 100)

    for i, v in enumerate(percentages):
        if i == 0:
            plt.text(0, percentages[i] + 1, "count:" + str(sentence_with_error), ha='center', va='bottom')
            plt.text(0, percentages[i] + 6, str("{:.3f}".format(percentages[i])) + '%', ha='center', va='bottom')
        else:
            plt.text(i, v + 1, "count:" + str(ngram_sentences_count[i]), ha='center', va='bottom')
            plt.text(i, v + 6, str("{:.3f}".format(percentages[i]))+'%', ha='center', va='bottom')
    plt.show()

    word_percentages = [0, 0, 0, 0]
    labels = ['1-gram', '2-gram', '3-gram', '4-gram']
    for i in ngram_counts:
        word_percentages[i - 1] = ngram_counts[i] / total_words * 100

    plt.bar(labels, word_percentages, color='green')
    plt.xlabel('Destructive N-gram Duplicates')
    plt.ylabel('Percentage %')
    plt.title('Percentage of Destructive N-gram Duplicates from total of ' + str(total_words) + 'words')
    plt.ylim(0, 100)
    for i, v in enumerate(word_percentages):
        plt.text(i, v + 1, "count:" + str(ngram_counts[i + 1]), ha='center', va='bottom')
        plt.text(i, v + 6, str("{:.3f}".format(word_percentages[i])) + '%', ha='center', va='bottom')
    plt.show()


def calculate_Type2_missing_glosses(data_array, return_sentences_with_error=False):
    missing_gloss_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_glosses = len(data_array)
    sentences_with_error = []

    for data in data_array:
        sentence = data["prediction"]
        prediction_glosses = set(re.split(r'\s+|[,\.]', data["prediction"]))
        reference_glosses = set(re.split(r'\s+|[,\.]', data["reference"]))

        #total_glosses += len(data["reference"])

        missing_glosses = reference_glosses - prediction_glosses
        missing_word_count = len(missing_glosses)

        if missing_word_count > 2:
            sentences_with_error.append(sentence)

        if missing_word_count in missing_gloss_counts:
            missing_gloss_counts[missing_word_count] += 1

    if return_sentences_with_error:
        return sentences_with_error

    labels = ['1-Missed', '2-Missed', '3-Missed', '4-Missed', '5-Missed']
    counts = [missing_gloss_counts.get(i, 0) for i in range(1, 6)]

    max_value_y = max(counts)
    plt.ylim(0, max_value_y + max_value_y / 5)
    plt.bar(labels, counts, color='purple')
    plt.xlabel('Number of Missed Glosses')
    plt.ylabel('Count')
    plt.title('Count of Missed Glosses from total of ' + str(total_glosses))
    for i, v in enumerate(counts):
        plt.text(i, v + 1, "count:" + str(counts[i]), ha='center', va='bottom')
        plt.text(i, v + max_value_y / 12, str("{:.3f}".format(counts[i] / total_glosses *100)) + '%', ha='center', va='bottom')
    plt.show()



def calculate_Type3_swapping_glosses(data_array,return_sentences_with_error=False):
    swapped_sentences_error = 0
    total_sentences = len(data_array)
    sentences_with_error = []

    for data in data_array:
        sentence = data["prediction"]
        prediction_glosses = re.split(r'\s+|[,\.]', data["prediction"])
        reference_glosses  = re.split(r'\s+|[,\.]', data["reference"])
        prediction_glosses_set = set(prediction_glosses)
        reference_glosses_set = set(reference_glosses)

        if len(prediction_glosses_set - reference_glosses_set) == 0:
            if prediction_glosses != reference_glosses:
                swapped_sentences_error += 1
                sentences_with_error.append(sentence)

    if return_sentences_with_error:
        return sentences_with_error

    print("Nr of sentences with swapped glosses: ", swapped_sentences_error)

    labels = ['Swapped Glosses']
    counts = swapped_sentences_error

    max_value_y = counts
    plt.ylim(0, max_value_y + max_value_y / 5)
    plt.bar(labels, counts, color='red')
    plt.xlabel('Number of Sentences with swapped Glosses')
    plt.ylabel('Count')
    plt.title('Count of Sentences with swapped Glosses from total of ' + str(total_sentences))
    plt.text(0, counts + 1, "count:" + str(counts), ha='center', va='bottom')
    plt.text(0, counts + max_value_y/8, str("{:.3f}".format(counts / total_sentences *100)) + '%', ha='center', va='bottom')
    plt.show()


def calculateType4_aditional_glosses(data_array, return_sentences_with_error=False):
    additional_gloss_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_sentences = len(data_array)
    sentences_with_error = []

    for data in data_array:
        sentence = data["prediction"]
        prediction_glosses = set(re.split(r'\s+|[,\.]', data["prediction"]))
        reference_glosses = set(re.split(r'\s+|[,\.]', data["reference"]))

        additional_glosses = prediction_glosses - reference_glosses
        additional_word_count = len(additional_glosses)

        if additional_word_count > 0 and additional_word_count < 3:
            sentences_with_error.append(sentence)

        if additional_word_count in additional_gloss_counts:
            additional_gloss_counts[additional_word_count] += 1

    if return_sentences_with_error:
        return sentences_with_error

    labels = ['1-Added', '2-Added', '3-Added', '4-Added', '5-Added']
    counts = [additional_gloss_counts.get(i, 0) for i in range(1, 6)]
    max_value_y = max(counts)
    plt.ylim(0, max_value_y + max_value_y/5)
    plt.bar(labels, counts, color='green')
    plt.xlabel('Number of additional Glosses')
    plt.ylabel('Count')
    plt.title('Count of Additional Glosses in sentences from total of ' + str(total_sentences) + " sentences")
    for i, v in enumerate(counts):
        plt.text(i, v + 1, "count:" + str(counts[i]), ha='center', va='bottom')
    plt.show()

def get_all_sentences(data_array):
    sentences = []
    for data in data_array:
        sentences.append(data["prediction"])
    return sentences
def remove_sentences_from(sentences, data_array):
    for sentence_to_remove in sentences:
        for data in data_array:
            if data["prediction"] == sentence_to_remove:
                data_array.remove(data)
                break
    return data_array
def calculate_overall_errors(data_array):
    total_number_of_sentences = len(data_array)
    all_sentences = data_array

    print("Total Sentences", len(all_sentences))

    type1_damaging_sentences = set(calculate_Type1_errors(data_array, "damaging", True, True))
    all_sentences = remove_sentences_from(type1_damaging_sentences, all_sentences)
    print("Errors of Type1 Damaging: ", len(type1_damaging_sentences), "Remained Sentences:", len(all_sentences))

    type2_missing_glosses = set(calculate_Type2_missing_glosses(all_sentences, True))
    all_sentences = remove_sentences_from(type2_missing_glosses, all_sentences)
    print("Errors of Type2: ", len(type2_missing_glosses), "Remained Sentences:", len(all_sentences))

    type3_swapping_glosses = set(calculate_Type3_swapping_glosses(all_sentences, True))
    all_sentences = remove_sentences_from(type3_swapping_glosses, all_sentences)
    print("Errors of Type3:", len(type3_swapping_glosses), "Remained Sentences:", len(all_sentences))

    type4_additional_glosses = set(calculateType4_aditional_glosses(all_sentences, True))
    all_sentences = remove_sentences_from(type4_additional_glosses, all_sentences)
    print("Errors of Type4: ", len(type4_additional_glosses), "Remained Sentences:", len(all_sentences))

    type1_additional_sentences = set(calculate_Type1_errors(all_sentences, "additional", False, True))
    all_sentences = remove_sentences_from(type1_additional_sentences, all_sentences)
    print("Errors of Type1 Additional: ", len(type1_additional_sentences), "Remained Sentences:", len(all_sentences))
    good_sentences_count = len(all_sentences)
    print("Good Sentences: ", good_sentences_count)
    error_counts = {
        "Type1 Damaging": len(type1_damaging_sentences),
        "Type2 Missing": len(type2_missing_glosses),
        "Type3 Swapping": len(type3_swapping_glosses),
        "Type4 Additional": len(type4_additional_glosses),
        "Type1 Repetitive Additional": len(type1_additional_sentences),
        "Good Sentences": good_sentences_count
    }
    error_percentages = {error_type: count / total_number_of_sentences * 100 for error_type, count in
                         error_counts.items()}

    # Plot the graph
    plt.bar(error_counts.keys(), error_counts.values(), color=['blue', 'orange', 'green', 'red', 'purple', 'cyan'])
    plt.xlabel('Error Types')
    plt.ylabel('Count')
    plt.title('Error Counts and Good Sentences from a total of ' + str(total_number_of_sentences) + ' sentences')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    max_value_y = max(error_counts.values())
    plt.ylim(0, max_value_y + max_value_y/5)
    for i, (error_type, count) in enumerate(error_counts.items()):
        plt.text(i, count + 1, f"Count: {count}", ha='center', va='bottom')
        plt.text(i, count + max_value_y/12, f"{error_percentages[error_type]:.2f}%", ha='center', va='bottom')

    plt.show()


def calculate_graphs(file_path):
    data = read_txt_file(file_path)
    calculate_Type1_repetitiveAditionale_avansat(data)
    calculate_Type1_repetitiveDaunatoare(data)
    calculate_Type2_missing_glosses(data)
    calculate_Type3_swapping_glosses(data)
    calculateType4_aditional_glosses(data)
    calculate_overall_errors(data)


calculate_graphs('./Data/GermanData.txt')
