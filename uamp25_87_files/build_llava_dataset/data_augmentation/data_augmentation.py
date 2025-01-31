import os
import sys
import cv2
import numpy as np
import random
from tqdm import tqdm


# go up one directory level from this file's directory:
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# prepend parent directory to the system path:
sys.path.insert(0, path)

from load_dataset import *

from uamp25_87_files.utils import get_augmentation_name, get_video_path

def horizontal_flip(frame, random_variable_assigned=None):
    return cv2.flip(frame, 1), random_variable_assigned


def invert_color(frame, random_variable_assigned=None):
    return cv2.bitwise_not(frame), random_variable_assigned


def random_rotate(frame, angle_range=(0, 50), random_variable_assigned=None):
    if random_variable_assigned == None:
        angle = random.uniform(*angle_range)
    else:
        angle = random_variable_assigned

    h, w = frame.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(frame, rotation_matrix, (w, h)), angle


def random_translate(frame, max_shift=(50, 50), random_variable_assigned=None):
    if random_variable_assigned == None:
        tx = 50
        ty = 50
    else:
        tx = random_variable_assigned[0]
        ty = random_variable_assigned[1]

    h, w = frame.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(frame, translation_matrix, (w, h)), (tx, ty)


def apply_augmentation(frame, technique_combo, random_variable_assigned):
    augmented_frame = frame.copy()
    if isinstance(technique_combo, list):
        for technique in technique_combo:
            augmented_frame, random_variable_assigned = technique(augmented_frame, random_variable_assigned=random_variable_assigned)

    else:
        augmented_frame, random_variable_assigned = technique_combo(augmented_frame, random_variable_assigned=random_variable_assigned)

    return augmented_frame, random_variable_assigned


def process_video(video_path, output_path, technique_combo, rows):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    random_variable_assigned = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply augmentations to the frame
        augmented_frame, random_variable_assigned = apply_augmentation(frame, technique_combo, random_variable_assigned)

        # Write the augmented frame to the output video
        out.write(augmented_frame)

    cap.release()
    out.release()


def process_videos(df, techniques, exercise):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        video_path = get_video_path(row["filename"], exercise)
        output_file_path = video_path.split(os.sep)
        for technique_combo in techniques:
            new_output_path = output_file_path.copy()
            techinique_name = get_augmentation_name(technique_combo)

            new_output_path[-2] = f"videos_{techinique_name}"
            new_output_path = os.path.join(*new_output_path)
            folder = os.path.dirname(new_output_path)
            if not os.path.exists(folder):
                os.makedirs(folder)

            if not os.path.isfile(new_output_path):
                rows = df[df["filename"] == row["filename"]]
                process_video(video_path, new_output_path, technique_combo, rows)

# if __name__ == "__main__":
#     video_files = ["archives_data/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/videos/32987_5.mp4"]  # Replace with your video paths
#     output_directory = "output_videos"  # Directory to save augmented videos

#     
#     augmentations = [
#         invert_color,
#         random_rotate,
#         horizontal_flip,
#         [horizontal_flip, invert_color],
#         [horizontal_flip, random_rotate],
#     ]

#     process_videos(video_files, output_directory, augmentations)
