import copy
import os.path
import random
import sys

from tqdm import tqdm
from textblob import Word

# go up one directory level from this file's directory:
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
# prepend parent directory to the system path:
sys.path.insert(0, path)

from load_dataset import *

from uamp25_87_files.utils import (
    get_augmentation_name,
    get_keypoint_path,
    get_video_path,
    path_keypoints,
    path_label_errors,
    set_all_seeds,
    video_id,
    possible_correct_messages,
    possible_only_one_error_messages,
    possible_multiple_errors_messages,
    human36m_body_parts,
    find_nearby_joints,
    get_body_parts_by_indexes
)
from uamp25_87_files.build_llava_dataset.data_augmentation.data_augmentation import invert_color, random_rotate, horizontal_flip, process_videos


set_all_seeds(42)
data_augmentation = [
        # invert_color,
        random_rotate,
        horizontal_flip,
        # [horizontal_flip, invert_color],
        # [horizontal_flip, random_rotate],
    ]
include_hpe_data = False
# Function to create error messages dynamically
def construct_error_message_boolean(row):
    messages = []
    for col in row.index:
        if col.startswith("error_type_"):
            error_name = col.replace("error_type_", "").replace("_", " ")
            if row[col] is not None:
                messages.append(
                    f"{error_name}"
                )
            # else:
            #     messages.append(f"{error_name}: None")
    if len(messages) == 0:

        # return "He performs it correctly, without any subtle errors."
        return random.choice(possible_correct_messages)
    elif len(messages) == 1:
        first_part_message = random.choice(possible_only_one_error_messages)
    elif len(messages) > 1:
        # messages = messages[::-1]
        first_part_message = random.choice(possible_multiple_errors_messages)
    messages = " and ".join(messages)
    resulting_message = f"{first_part_message} {messages}."
    return resulting_message


def construct_error_message_temporal(row):
    messages = []
    for col in row.index:
        if col.startswith("error_type_"):
            error_name = col.replace("error_type_", "").replace("_", " ")
            if row[col] is not None:
                if isinstance(row[col], list):
                    continue
                else:
                    
                    messages.append(f"{error_name} from frame time {row[col][0]}s to {row[col][1]}s")
            # else:
            #     messages.append(f"{error_name}: None")
    if len(messages) == 0:

        # return "He performs it correctly, without any subtle errors."
        return random.choice(possible_correct_messages)
    elif len(messages) == 1:
        first_part_message = random.choice(possible_only_one_error_messages)
    elif len(messages) > 1:
        # messages = messages[::-1]
        first_part_message = random.choice(possible_multiple_errors_messages)
    messages = " and ".join(messages)
    resulting_message = f"{first_part_message} {messages}."
    return resulting_message

def filter_frames(data, max_N_frames):
    total = len(data)

    if total <= max_N_frames:
        return data
    # Compute equally spaced indices
    indices = [round(i * (total - 1) / (max_N_frames - 1)) for i in range(max_N_frames)]
    assert len(indices) == max_N_frames

    return [data[i] for i in indices]

def filter_keypoints(hpe_data, relevant_keypoints):
    for data in hpe_data:
        for data_instances in data["instances"]:
            data_instances["keypoints"] = [keypoint for i, keypoint in enumerate(data_instances["keypoints"]) if i in relevant_keypoints]
    return hpe_data


def hpe_data_message(row):
    with open(get_keypoint_path(row['filename'], exercise)) as f:
        hpe_data = json.load(f)

    for data in hpe_data:
        for data_instances in data["instances"]:
            del data_instances["keypoint_scores"]

    # get error type
    error_body_part = []
    relevant_keypoints = set()
    for col in row.index:
        if col.startswith("error_type_"):
            error_name = col.replace("error_type_", "").replace("_", " ")
            error_single_word = error_name.split(" ")
            for word in error_single_word:
                word = Word(word).singularize()
                for key, value in human36m_body_parts.items():
                    if word.lower() in (key.lower()):
                        for index in value:
                            relevant_keypoints.add(index)

    relevant_keypoints = find_nearby_joints(relevant_keypoints, 1)
    relevant_keypoints = sorted(relevant_keypoints)
    # reduce the frames number
    hpe_data = filter_frames(hpe_data, 16)
    # reduce the keypoints cardinality
    hpe_data = filter_keypoints(hpe_data, relevant_keypoints)
    relevant_bodyparts = get_body_parts_by_indexes(relevant_keypoints)
    # relevant_bodyparts_to_string = ", ".join(relevant_bodyparts)

    return f"To help you identify the subject, these are the Human Pose Estimation data extract in a 3D space, for the following body parts in the following order {relevant_bodyparts}: {hpe_data}."

# Function to create the JSON structure for each row
def create_json_structure_llava(row, idx, data_type=None, augmented=False):
    video_path = get_video_path(row["filename"], exercise).replace("archives_data/", "")
    # if data_augmentation is not None:
    #     print(augmented)
    #     video_path = video_path.replace("vicdeos", "videos_horizontal_flip")

    if data_type is not None:
        data_source= f"Fitness-AQA_{data_type}",   
    else: 
        data_source= "Fitness-AQA",    

    input_message = (
        f"<image>\nThe subject is performing a Squat or a Overhead Press (OHP) exercise. Which one is he making? Is he making a mistake? If so, what mistake is he making?"
    )
    if include_hpe_data:
        input_message = f"{input_message} {hpe_data_message(row)}."
    return {
        # "id": f"{video_id(video_path)}",
        "id": row["filename"],
        "conversations": [
            {"from": "human", "value": input_message},
            {"from": "gpt", "value": f"The subject is performing a {exercise}. {construct_error_message_boolean(row)}"},
        ],
        "data_source": data_source,
        "video": video_path,
    }

def create_json_structure_qwen(row, idx, data_type=None, augmented=False):
    video_path = get_video_path(row["filename"], exercise).replace("archives_data/", "")
    # if data_augmentation is not None:
    #     print(augmented)
    #     video_path = video_path.replace("videos", "videos_horizontal_flip")

    if data_type is not None:
        data_source = (f"Fitness-AQA_{data_type}",)
    else:
        data_source = ("Fitness-AQA",)

    input_message = f"The subject is performing a Squat or a Overhead Press (OHP) exercise. Which one is he making? Is he making a mistake? If so, what mistake is he making?"
    if include_hpe_data:
        input_message = f"{input_message} {hpe_data_message(row)}."
    return {
        # "id": f"{video_id(video_path)}",
        "id": row["filename"],
        "conversations": [
            {"from": "user", "value": input_message},
            {"from": "assistant", "value": f"The subject is performing a {exercise}. {construct_error_message_boolean(row)}"},
        ],
        "data_source": data_source,
        "video": video_path,
    }

def get_filenames_with_only_specific_error(df, error_1, error_2, filter_type="not"):

    error_1_filenames = set(df[(df["error_type"] == error_1) & (df["label"].notnull())]["filename"])

    error_2_filenames = set(df[(df["error_type"] == error_2) & (df["label"].notnull())]["filename"])

    if filter_type == "not":
        unique_filenames = error_1_filenames - error_2_filenames
    elif filter_type == "and":
        unique_filenames = error_1_filenames & error_2_filenames

    return unique_filenames


def get_errors_count(df_train, print_counts=False):
    # get the balance of the dataset for each error_type inside
    error_types = sorted(df_train["error_type"].unique().tolist())

    file_with_error_0_only = get_filenames_with_only_specific_error(df_train, error_types[0], error_types[1], "not")
    file_with_error_1_only = get_filenames_with_only_specific_error(df_train, error_types[1], error_types[0], "not")
    file_with_both_errors = get_filenames_with_only_specific_error(df_train, error_types[0], error_types[1], "and")
    file_without_both_errors = set(df_train["filename"]) - file_with_both_errors - file_with_error_0_only - file_with_error_1_only

    if print_counts:       
        print(f"label balances - \n{error_types[0]}: {len(file_with_error_0_only)}, \n{error_types[1]}: {len(file_with_error_1_only)}, \n{error_types[0]}&{error_types[1]}: {len(file_with_both_errors)}, \nNo errors: {len(file_without_both_errors)}")

    return {error_types[0]: file_with_error_0_only, error_types[1]: file_with_error_1_only, f"{error_types[0]}&{error_types[1]}": file_with_both_errors, "none": file_without_both_errors}


SEPARATOR = os.sep
exercise_type = ["Squat", "OHP"]

train_json_base, train_json_augmented, test_json = [], [], []

dataset_dict = {}
for exercise in exercise_type:
    # Load dataset
    dataset = LoadDatasetKeypoints()
    path_label = path_label_errors(exercise)
    df = dataset.load_dataset_info(path_keypoints, path_label)
    # max_lenght_sequence = dataset.get_max_lenght_sequence(df)
    # df_augmented = dataset.load_dataset_info(path_keypoints_augmented)
    # df_augmented = df_augmented[df_augmented["label"] == "[correct_posture]"]
    # df = pd.concat([df, df_augmented], ignore_index=True)

    if data_augmentation is not None:
        process_videos(df, data_augmentation, exercise)

    df_train, df_val, def_test = split_dataset(df, exercise)
    df_train = pd.concat([df_train, df_val], ignore_index=True)
    del df_val

    balance = get_errors_count(df_train)

    dataset_dict[exercise] = {"dataset": {"df_train": df_train, "df_test": def_test}, "balance": balance}


def create_json_for_llava_training(df: pd.DataFrame, data_type=None, balance=None, min=None):
    original_balance = copy.deepcopy(balance)
    df = df.pivot(index="filename", columns="error_type", values="label").reset_index()
    df.columns = [f"error_type_{col}" if col != "filename" else col for col in df.columns]

    json_list = [create_json_structure_llava(row, idx, data_type) for idx, row in df.iterrows()]
    if balance is None:
        return json_list
    # new_json_list = copy.deepcopy(json_list)

    if data_augmentation is not None and balance is not None and min is not None:
        # get the max lenght of files in balance
        # max_lenght = max([len(files) for files in balance.values()])

        for key, value in balance.items():
            files = value
            files = list(files)
            if key == "none":
                files = set(files[:min])
            else:
                files = set(files[:min])
            balance[key] = files

        new_json_list = [elem for elem in json_list if elem["id"] in set.union(*balance.values())]

        # return the ratio between the lenght of the files and the max lenght, for each key
        # balance = {key: {"files": files, "ratio": max_lenght / len(files)} for key, files in balance.items()}

        # add n augmented version of the files in balance equal to the ratio
        # the exceed of integer part of the ratio is added randomly, where the decimal part equals to the percentage of file to add
        # extract the list of files with a specific error from the json_list

        def add_augmented_file(json_list, technique_combo, ratio=1):
            techinique_name = get_augmentation_name(technique_combo)
            temp_list = []
            for elem in json_list:
                if elem["id"] in files:
                    temp = copy.deepcopy(elem)
                    temp["video"] = temp["video"].replace("videos", f"videos_{techinique_name}")
                    temp_list.append(temp)
            # if ratio < 1:
            #     temp_list = temp_list[: int(ratio * len(temp_list))]
            return temp_list

        for key, value in balance.items():
            files = value
            # ratio = value["ratio"]
            # data_augmentation_types_to_add = int(ratio) - 1  # -1 because the original files are already in the list
            # decimal_part = ratio - int(ratio)
            for i, technique_combo in enumerate(data_augmentation):
                # if i >= data_augmentation_types_to_add:
                #     break
                new_json_list.extend(add_augmented_file(json_list, technique_combo))
            # if decimal_part > 0 and i <= data_augmentation_types_to_add and i + 1 < len(data_augmentation):
            #     new_json_list.extend(add_augmented_file(json_list, data_augmentation[i + 1], decimal_part))

    # print final balance of the dataset
    if balance is not None:
        print(f"Balance of the dataset for {data_type}: before balancing")
        for key, value in original_balance.items():
            print(f"{key}: {len(value)}")
        print(f"Balance of the dataset for {data_type}: after balancing")
        for key, value in balance.items():
            files = value
            counter = 0
            for i, elem in enumerate(new_json_list):
                if elem["id"] in files:
                    counter += 1
            print(f"{key}: {counter}")
    return json_list, new_json_list

# get max lenght of sample for error_type
max = 0
min = 100000
for exercise, value in dataset_dict.items():
    for value in value["balance"].values():
        lenght = len(value)
        if lenght > max:
            max = lenght
        if lenght < min:
            min = lenght

for exercise, value in dataset_dict.items():

    balance = value['balance']
    df_train = value['dataset']['df_train']
    df_test = value['dataset']['df_test']

    temp_train_base, temp_train_augmented = create_json_for_llava_training(df_train, data_type=f"{exercise}_train", balance=balance, min=min)
    train_json_base.extend(temp_train_base)
    train_json_augmented.extend(temp_train_augmented)
    # with open(os.path.join(os.path.dirname(__file__), f"train_{exercise}.json"), "w") as json_file:
    #     json.dump(temp_train, json_file, indent=2)

    # val_json.extend(create_json_for_llava_training(df_val, data_type=f"{exercise}_val"))
    temp_test = create_json_for_llava_training(df_test, data_type=f"{exercise}_test")
    test_json.extend(temp_test)
    # with open(os.path.join(os.path.dirname(__file__), f"test_{exercise}.json"), "w") as json_file:
    #     json.dump(temp_test, json_file, indent=2)

with open(os.path.join(os.path.dirname(__file__), "train_step1.json"), "w") as json_file:
    json.dump(train_json_base, json_file, indent=2)

with open(os.path.join(os.path.dirname(__file__), "train_step2.json"), "w") as json_file:
    json.dump(train_json_augmented, json_file, indent=2)

# with open(os.path.join(os.path.dirname(__file__), "val.json"), "w") as json_file:
#     json.dump(val_json, json_file, indent=2)

with open(os.path.join(os.path.dirname(__file__), "test.json"), "w") as json_file:
    json.dump(test_json, json_file, indent=2)
