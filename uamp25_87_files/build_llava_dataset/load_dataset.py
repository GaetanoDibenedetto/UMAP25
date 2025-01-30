import json
import os
import pandas as pd
import re
import os
import numpy as np
from uamp25_87_files.utils import path_split_info, get_keypoints_from_filename

class LoadDatasetKeypoints():
    def __init__(self, keypoint_path = '', dataframes_path = '', make_framing_from_video = True, pad_lenght_wanted = 1, action_to_remove = []):
        self.keypoint_path = keypoint_path

        pass

    def load_dataset_info(self,
        keypoint_path, label_path=None):

        if label_path!=None:
            self.label_path = label_path

            # initialize the dataframe with a empty dataframe
            self.df_label = pd.DataFrame().assign(error_type=[])

            for file, file_path in label_path.items():
                with open(file_path) as f:
                    data = json.load(f)
                temp_df = pd.DataFrame.from_dict(data, orient="index").assign(error_type=file).rename_axis("filename").reset_index()
                self.df_label = pd.concat([self.df_label, temp_df])

        columns_with_error_values = [col for col in self.df_label.columns if isinstance(col, int)]
        # Extract min start time and max end time, handling None and NaN
        def calculate_min_max(row):
            # Ensure row is iterable and filter out None or NaN values
            valid_values = []
            for x in row:
                if x is not None:
                    if type(x) is not list:
                        continue
                    if pd.isna(x).all() is False:
                        continue
                    valid_values.append(x)

            # If valid values exist, calculate min and max
            if valid_values:
                min_start_time = min(x[0] for x in valid_values)
                max_end_time = max(x[1] for x in valid_values)
                return (min_start_time, max_end_time)
            else:  # No valid values
                return None

        self.df_label["label"] = self.df_label[columns_with_error_values].apply(calculate_min_max, axis=1)
        self.df_label = self.df_label.drop(columns=columns_with_error_values)

        return self.df_label

    def get_max_lenght_sequence(self, df):
        max_lenght = df["filename"].apply(get_lenght_sequence_from_keypoints).max()
        return max_lenght

def get_lenght_sequence_from_keypoints(filename):
    keypoints = get_keypoints_from_filename(filename)
    return len(keypoints)

def split_dataset(df, exercise_type):
    with open(path_split_info(exercise_type)["train"]) as f:
        train_data = json.load(f)

    with open(path_split_info(exercise_type)["test"]) as f:
        test_data = json.load(f)

    with open(path_split_info(exercise_type)["val"]) as f:
        val_data = json.load(f)

    df_train = df[df["filename"].isin(train_data)]
    df_test = df[df["filename"].isin(test_data)]
    df_val = df[df["filename"].isin(val_data)]

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    assert len(df) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


def balance_dataset(df):
    # remove the label that ecced the counter of the lower one
    label_count = df["label_id"].value_counts()
    min_label_count = label_count.min()
    df.loc[:, "label_id_str"] = df["label_id"].astype(str)
    df = df.groupby("label_id_str").sample(min_label_count)
    label_count = df["label_id"].value_counts()
    return df
