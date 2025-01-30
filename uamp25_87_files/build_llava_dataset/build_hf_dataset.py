import os.path
import sys

# go up one directory level from this file's directory:
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
# prepend parent directory to the system path:
sys.path.insert(0, path)

from load_dataset import *

from uamp25_87_files.utils import get_video_path, path_keypoints, path_label_errors, set_all_seeds


set_all_seeds(42)
# Function to create error messages dynamically
def construct_error_message(row):
    messages = []
    for col in row.index:
        if col.startswith("error_type_"):
            error_name = col.replace("error_type_", "").replace("_", " ")
            if row[col] is not None:
                messages.append(
                    f"{error_name} error: from min {row[col][0]:.2f} to min {row[col][1]:.2f}"
                )
            else:
                messages.append(f"No {error_name} error found")
    return " - ".join(messages)

# Function to create the JSON structure for each row
def create_json_structure(row, idx, data_type=None):
    if data_type is not None:
        data_source= f"Fitness-AQA_{data_type}",   
    else: 
        data_source= "Fitness-AQA",    
    return {
        "id": f"{idx}",
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The subject is performing a {exercise}. Is he making a mistake? If so, what mistake is he making and when?"},
                    {"type": "video"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": construct_error_message(row)},
                ],
            },
        ],
        "data_source": data_source,
        "video": get_video_path(row["filename"], exercise).replace("archives_data/", ""),
    }

SEPARATOR = os.sep
exercise_type = ["Squat", "OHP"]

train_json, val_json, test_json = [], [], []

for exercise in exercise_type:
    # Load dataset
    dataset = LoadDatasetKeypoints()
    path_label = path_label_errors(exercise)
    df = dataset.load_dataset_info(path_keypoints, path_label)
    # max_lenght_sequence = dataset.get_max_lenght_sequence(df)
    # df_augmented = dataset.load_dataset_info(path_keypoints_augmented)
    # df_augmented = df_augmented[df_augmented["label"] == "[correct_posture]"]
    # df = pd.concat([df, df_augmented], ignore_index=True)

    df_train, df_val, def_test = split_dataset(df, exercise)

    train_label_balance = len(df_val["label"] == None) / len(df_train)
    test_label_balance = len(def_test["label"] == None) / len(df_train)
    val_label_balance = len(df_val["label"] == None) / len(df_train)

    def create_json_for_llava_training(df: pd.DataFrame, data_type=None):
        df = df.pivot(index="filename", columns="error_type", values="label").reset_index()
        df.columns = [
            f"error_type_{col}" if col != "filename" else col for col in df.columns
        ]
        # df['keypoints'] = df["filename"].apply(get_keypoints_from_filename)
        # df['datasource'] = 'Fitness-AQA',
        # df['conversation'] = list(dict("from":"human", "value":"<image>\nWhat is the mistake committed by this man and when was it committed?"),
        #                      dict("from":"gpt", "value":f"{{col}" if col != "filename" else col for col in df.columns}))

        json_list = [create_json_structure(row, idx, data_type) for idx, row in df.iterrows()]

        return json_list

    train_json.extend(create_json_for_llava_training(df_train, data_type=f"{exercise}_train"))

    val_json.extend(create_json_for_llava_training(df_val, data_type=f"{exercise}_val"))

    test_json.extend(create_json_for_llava_training(def_test, data_type=f"{exercise}_test"))


with open(os.path.join(os.path.dirname(__file__), "train.json"), "w") as json_file:
    json.dump(train_json, json_file, indent=2)

with open(os.path.join(os.path.dirname(__file__), "val.json"), "w") as json_file:
    json.dump(val_json, json_file, indent=2)

with open(os.path.join(os.path.dirname(__file__), "test.json"), "w") as json_file:
    json.dump(test_json, json_file, indent=2)
