import os
import random
import numpy as np
import torch
import json
from decord import VideoReader, cpu
import sys
# import ffmpeg

data_archives_path = "archives_data/Fitness-AQA_dataset_release"
output_predictions_path = "inference_output"

path_keypoints = lambda exercise_type: os.path.join(data_archives_path, exercise_type, "motion_bert", "predictions")

path_video = lambda exercise_type: os.path.join(data_archives_path, exercise_type, "Labeled_Dataset", "videos")

path_label_errors = lambda exercise_type: get_path_label_errors(exercise_type)

video_id = lambda video_path: video_path.split("/")[-1].replace(".mp4", '')

#     if exercise_type == "Squat":
#         dict(
#     knees_forward=os.path.join(data_archives_path, exercise_type, "Labeled_Dataset", "Labels", "error_knees_forward.json"),
#     knees_inward=os.path.join(data_archives_path, exercise_type, "Labeled_Dataset", "Labels", "error_knees_inward.json")
# )
#     elif exercise_type == "OHP":
#         dict(
#     knees_forward=os.path.join(data_archives_path, exercise_type, "Labeled_Dataset", "Labels", "error_knees_forward.json"),
#     knees_inward=os.path.join(data_archives_path, exercise_type, "Labeled_Dataset", "Labels", "error_knees_inward.json")
# )

path_split_info = lambda exercise_type: dict(
    train = os.path.join(data_archives_path, exercise_type, "Labeled_Dataset", "Splits", "train_keys.json"),
    test = os.path.join(data_archives_path, exercise_type, "Labeled_Dataset", "Splits", "test_keys.json"),
    val = os.path.join(data_archives_path, exercise_type, "Labeled_Dataset", "Splits", "val_keys.json")
    )

def get_path_label_errors(exercise_type):
    files = os.listdir(os.path.join(data_archives_path, exercise_type, "Labeled_Dataset", "Labels"))
    # only files which end with .json
    json_files = [file for file in files if file.endswith(".json")]
    dict_errors = {}
    for error_file in json_files:
        error_name = error_file.replace(".json", '').split("_")
        error_name = "_".join(error_name[1:])
        dict_errors[error_name] = os.path.join(data_archives_path, exercise_type, "Labeled_Dataset", "Labels", error_file)

    return dict_errors

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_keypoint_path(filename, exercise_type):
    keypoint_path = os.path.join(path_keypoints(exercise_type), filename + '.json')
    return keypoint_path

def get_video_path(filename, exercise_type):
    video_path = os.path.join(path_video(exercise_type), filename + ".mp4")
    return video_path

def load_keypoint(keypoint_path, normalize_keypoint=False):
    keypoint = np.load(keypoint_path, allow_pickle=True)
    keypoint = np.array([keypoint])
    keypoint = torch.tensor(keypoint).to(torch.float32)
    return keypoint


def load_keypoint(keypoint_path, normalize_keypoint=True):

    data = json.load(open(keypoint_path))
    keypoints = []
    for elem in data:
        data_for_each_frame = elem['instances']
        # check multiple subjects?
        assert len(data_for_each_frame) == 1
        keypoint_for_frame = data_for_each_frame[0]['keypoints']
        keypoints.append(keypoint_for_frame)

    return load_keypoint_in_torch(keypoints)


def load_keypoint_in_torch(keypoint):
    keypoint = np.array([keypoint])
    keypoint = torch.tensor(keypoint).to(torch.float32)
    return keypoint


# def normalize(volume):
#     """Normalize the volume"""
#     # scale in a 0-1 range
#     volume = (volume - torch.min(volume)) / max(
#         (torch.max(volume) - torch.min(volume)), 1
#     )
#     return volume.to(torch.float32)


def one_hot_encode(label, num_classes=2):
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


def get_keypoints_from_filename(filename):
    keypoint_path = get_keypoint_path(filename)
    return load_keypoint(keypoint_path).squeeze()


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames, frame_time, video_time

# def horizontal_flip_video(input_path, output_path):
#     # check if the folder where i want to save the video exists
#     folder = os.path.dirname(output_path)
#     if not os.path.exists(folder):
#         os.makedirs(folder)

#     stream = ffmpeg.input(input_path)
#     stream = ffmpeg.hflip(stream)
#     stream = ffmpeg.output(stream, output_path, loglevel="quiet")
#     ffmpeg.run(stream)

def get_augmentation_name(technique_combo):
    if isinstance(technique_combo, list):
        techinique_name = "&".join([tech.__name__ for tech in technique_combo])
    else:
        techinique_name = technique_combo.__name__
    return techinique_name

possible_correct_messages = [
    "He performs it correctly, without any subtle errors.",
    "He performs it correctly, without any errors.",
    "The exercise is executed correctly, with no errors observed.",
    "The subject performs the exercise flawlessly, without any mistakes.",
    "No errors are detected in the performance of this exercise.",
    "The execution is correct, with no visible errors.",
    "The subject completes the exercise properly, without any noticeable errors.",
    "The performance is accurate, with no mistakes identified.",
    "The exercise is carried out correctly, free of any errors.",
    "No mistakes are present in the execution of this exercise.",
    "The subject performs the movement correctly, without any faults.",
    "The action is performed without error, demonstrating proper execution.",
]

possible_only_one_error_messages = [
    "The body part involved in the error is as follows:",
    "The following body part is responsible for the error:",
    "An error is identified in the following body part:",
    "The body part exhibiting the error is as follows:",
    "The following body part is committing the error:",
    "The error is associated with the following body part:",
    "The specific body part with an error is as follows:",
    "The body part displaying the error is as follows:",
    "The following is the body part where the error occurred:",
    "The error involves the following body part:",
    "The body part committing errors is the following:",
    "The body part committing an error is the following:",
]

possible_multiple_errors_messages = [
    "The body parts committing errors are as follows:",
    "Errors are observed in the following body parts:",
    "The following body parts are responsible for the errors:",
    "The body parts where errors are detected are as follows:",
    "Errors involve the following body parts:",
    "The following body parts exhibit errors:",
    "Mistakes are identified in the following body parts:",
    "The errors are associated with these body parts:",
    "The following body parts are committing errors:",
    "The specific body parts with errors are as follows:",
]


def compute_iou(pred_segment, gt_segment):
    """
    Compute the Intersection over Union (IoU) between two temporal segments.
    Args:
        pred_segment (list or tuple): [start_time, end_time] of the predicted segment.
        gt_segment (list or tuple): [start_time, end_time] of the ground truth segment.
    Returns:
        float: IoU value.
    """
    pred_segment = pred_segment if pred_segment is not None else [0, 0]
    gt_segment = gt_segment if gt_segment is not None else [0, 0]
    
    inter_start = max(pred_segment[0], gt_segment[0])
    inter_end = min(pred_segment[1], gt_segment[1])
    
    intersection = max(0, abs(inter_end - inter_start))

    union = (pred_segment[1] - pred_segment[0]) + (gt_segment[1] - gt_segment[0]) - intersection
    return intersection / union if union > 0 else 0


def compute_average_precision(predictions, ground_truths, iou_threshold):
    """
    Compute the Average Precision (AP) at a specific IoU threshold.
    Args:
        predictions (list of tuples): List of predicted segments [(start_time, end_time)].
        ground_truths (list of tuples): List of ground truth segments [(start_time, end_time)].
        iou_threshold (float): IoU threshold to consider a prediction correct.
    Returns:
        float: Average Precision (AP) score.
    """
    assert len(predictions) == len(ground_truths), "Predictions and ground truths must have the same length."

    correct_predicted_proposal = 0

    for pred_segment, gt_segment in zip(predictions, ground_truths):
        iou = compute_iou(pred_segment, gt_segment)
        if iou >= iou_threshold:
            correct_predicted_proposal += 1

    # , Precision (P) for class (c) in a single video is calculated by: P(c) = sum(correct predicted proposals) / sum(predicted proposals)

    precision = correct_predicted_proposal / (len(predictions) + 1e-6)

    # Average Precision = for all videos in the test set is calculated by sum(P(c)) / All videos in testing set for class c

    average_precision = precision
    return average_precision


def compute_ap_score(y_true, y_pred, iou_thresholds=[0.5]):
    """
    Compute mean Average Precision (mAP) over multiple IoU thresholds.
    Args:
        predictions (list of tuples): List of predicted segments [(start_time, end_time)].
        ground_truths (list of tuples): List of ground truth segments [(start_time, end_time)].
        iou_thresholds (list of floats): List of IoU thresholds.
    Returns:
        float: mAP score.
    """
    # mean Average Precision (mAP) for all classes is calculated by: sum(AP for all classes) / sum (/)number of videos)
    aps = []
    for iou_threshold in iou_thresholds:
        ap = compute_average_precision(y_pred, y_true, iou_threshold)
        aps.append(ap)
    return np.mean(aps)

from collections import deque


def build_skeleton_graph():
    """MMpose - Human3.6 representation"""
    skeleton = {
        10: [9],  # Head → Nose
        9: [8],  # Nose → Neck
        8: [14, 11, 7],  # Neck → Shoulders & Spine
        14: [15],  # Right Shoulder → Right Elbow
        11: [12],  # Left Shoulder → Left Elbow
        15: [16],  # Right Elbow → Right Wrist
        12: [13],  # Left Elbow → Left Wrist
        7: [0],  # Spine → Mid Hip/Pelvis
        0: [4, 1],  # Mid Hip/Pelvis → Hips
        4: [2],  # Right Hip → Right Knee
        1: [5],  # Left Hip → Left Knee
        2: [3],  # Right Knee → Right Ankle
        5: [6],  # Left Knee → Left Ankle
    }
    return skeleton


# def bfs_nerby_joints(starting_joints, distance):
#     """Finds joints within a certain distance from the given starting joints with beam first search technique"""
#     skeleton = build_skeleton_graph()
#     visited = set()
#     queue = deque()

#     # Initialize queue with starting joints at distance 0
#     for joint in starting_joints:
#         queue.append((joint, 0))
#         visited.add(joint)

#     result = set()

#     while queue:
#         joint, dist = queue.popleft()

#         if dist <= distance:
#             result.add(joint)

#         if dist < distance:  # Continue exploring within distance limit
#             for neighbor in skeleton.get(joint, []):
#                 if neighbor not in visited:
#                     visited.add(neighbor)
#                     queue.append((neighbor, dist + 1))

#     return sorted(result)


def find_nearby_joints(start_joints, max_distance):
    skeleton = build_skeleton_graph()
    inverse_skeleton = {}
    for joint, neighbors in skeleton.items():
        for neighbor in neighbors:
            if neighbor not in inverse_skeleton:
                inverse_skeleton[neighbor] = []
            inverse_skeleton[neighbor].append(joint)

    visited = set()  # To track visited joints
    nearby_joints = set()  # To store the nearby joints
    queue = deque([(joint, 0) for joint in start_joints])  # Queue to manage BFS (joint, current distance)

    while queue:
        current_joint, distance = queue.popleft()

        # If this joint has been visited, skip it
        if current_joint in visited:
            continue

        # Mark the current joint as visited
        visited.add(current_joint)

        # If the current distance is within the max_distance, add it to nearby joints
        if distance <= max_distance:
            nearby_joints.add(current_joint)

            # Add neighbors in the correct direction (forward links)
            for neighbor in skeleton.get(current_joint, []):  # Forward direction
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

            # Add neighbors in the inverse direction (reverse links)
            for inverse_neighbor in inverse_skeleton.get(current_joint, []):  # Reverse direction
                if inverse_neighbor not in visited:
                    queue.append((inverse_neighbor, distance + 1))

    return nearby_joints


human36m_body_parts = {"Hip": [1, 0, 4], "Pelvis": [0], "Knee": [2, 5], "Ankle": [3, 6], "Spine": [7], "Neck": [8], "Nose": [9], "Head": [10], "Shoulder": [14, 11], "Elbow": [15, 12], "Wrist": [13, 16]}

def get_body_parts_by_indexes(indexes):   
    # Reverse the dictionary to create a mapping from index to body part
    index_to_part = {}
    for body_part, indices in human36m_body_parts.items():
        for index in indices:
            index_to_part[index] = body_part

    # Use the index_to_part mapping to find the body parts corresponding to the given indexes
    body_parts = []
    for index in indexes:
        if index in index_to_part:
            body_parts.append(index_to_part[index])

    return body_parts    
