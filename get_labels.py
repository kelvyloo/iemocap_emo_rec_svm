#!/usr/bin/python3

import os, csv

path_to_categories = "/home/kelvin/c487/data/labels/Categorical/"
wav_file_paths     = "/home/kelvin/c487/data/labels/wav_files_more.txt"
iemocap_filepath   = "/home/kelvin/c487/data/IEMOCAP_full_release/"
ground_truth_file  = "/home/kelvin/c487/data/ground_truth_more.csv"

categorical = os.listdir(path_to_categories)
categorical.sort()

count      = 0
label_dict = {}
emotions   = {}
label_enum = {'Neutral state' : 1,
              'Fear'          : 2,
              'Sadness'       : 2,
              'Anger'         : 3,
              'Disgust'       : 3,
              'Frustration'   : 3,
              'Excited'       : 4,
              'Surprise'      : 4,
              'Happiness'     : 4}

def label_to_enum(value):
    emotion_string = value[0]

    try:
        label = [emotion_string, label_enum[emotion_string]]
    except:
        label = [emotion_string, 0]

    return label

def check_label(emotion_label):

    if not emotion_label[1]:
        return False

    return True

# Get all wav file names and labels into a dictionary
for category_file in categorical:
    # Open category text files
    if (category_file[-3:] == 'txt'):
        category_file = open(path_to_categories+category_file,"r")
        try:
            lines = category_file.readlines()
        except:
            category_file.close

        # Create dictionary entry:
        #   key = wav file name
        #   value = labels
        for line in lines:
            line = line.split(':')
            key = line[0].strip(' ')

            value = line[1].split(';')
            value = value[0:-1]

            if key not in label_dict:
                label_dict[key] = value

# Write full wav file paths to txt file
with open(wav_file_paths, 'w') as out:
    for key, value in label_dict.items():
        # Check value is valid
        label = label_to_enum(value)

        if not check_label(label):
            continue

        # Build new dict with only valid entries
        if key not in emotions:
            emotions[key] = label

        # Build wav file path
        session_number = key[4]
        wav_folder = key[0:-5]

        wav_path = iemocap_filepath +           \
                   "Session" + session_number + \
                   "/sentences/wav" +           \
                   "/" + wav_folder +           \
                   "/" + key + '.wav\n'

        out.write(wav_path)

# Write ground truth labels to csv file
with open(ground_truth_file, 'w') as csv_f:
    label_writer = csv.writer(csv_f, delimiter=',')
    label_writer.writerow(['Wav file', 'Emotion Label', 'Emotion Enumeration'])

    for key, value in emotions.items():
        label_writer.writerow([key, value[0], value[1]])
