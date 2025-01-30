import json
import os
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import torch
import warnings
import numpy as np
from uamp25_87_files.utils import compute_ap_score, get_keypoint_path, load_video, output_predictions_path, possible_correct_messages, set_all_seeds
from tqdm import tqdm
import datetime


warnings.filterwarnings("ignore")
dataset_path = "archives_data"
set_all_seeds(42)


# Load the JSON file
def load_json_file(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def extract_exercise_type(sentence):  # Regular expression to extract the exercise type
    pattern = r"The subject is performing a ([^\.]+)\."

    match = re.search(pattern, sentence)
    if match:
        return match.group(1)
    else:
        return None


def extract_error_details_old(sentence):  # Regular expression to extract the error details
    # Split the sentence into individual error segments
    error_segments = sentence.split(" - ")
    errors = {}

    # Regular expressions to identify error patterns
    error_pattern = r"(?P<error_name>[a-zA-Z\s]+error)"
    time_range_pattern = r"from min (?P<start>\d+\.\d+) to min (?P<end>\d+\.\d+)"

    for segment in error_segments:
        # Match the error name
        error_name_match = re.search(error_pattern, segment)
        if error_name_match:
            error_name = error_name_match.group("error_name").strip()

            # Match the time range if present
            time_range_match = re.search(time_range_pattern, segment)
            if time_range_match:
                time_range = (float(time_range_match.group("start")), float(time_range_match.group("end")))
            else:
                time_range = None
                error_name = error_name.replace("No ", "").strip()

            errors[error_name] = time_range

    return errors


def error_parser(sentence, temporal=False):
    # Case 1: Two errors

    if temporal == False:
        match = re.search(r":\s*(.*)\s+and\s*([^.]*)", sentence)
    else:
        match = re.search(r":\s*(.*)\s+and\s+(.*)", sentence)

    if match:
        return {"errors": [match.group(1), match.group(2)]}

    # Case 2: One error
    if temporal == False:
        match = re.search(r":\s*([^.]*)", sentence)
    else:
        match = re.search(r":\s*(.*)", sentence)

    if match:
        return {"errors": [match.group(1)]}

    # Case 3: No errors
    if [result for result in possible_correct_messages if re.search(sentence, result)] != []:
        return {"errors": None}

    # Default case: Unrecognized format
    raise Exception("Wrong sentence format in the output?")


def extract_error_details_boolean(sentence):  # Regular expression to extract the error details
    # Split the sentence into individual error segments
    sentence = sentence.split(". ")
    sentence = [item for item in sentence if item != ""]

    assert len(sentence) == 2
    exercise_type_sentence = sentence[0].lstrip()
    error_details_sentence = sentence[1].lstrip()

    exercise_type_pattern = r"The subject is performing a ([A-Za-z]+)"
    exercise_type_match = re.search(exercise_type_pattern, exercise_type_sentence)
    exercise_type = exercise_type_match.group(1)

    results = {}
    if exercise_type == "Squat":
        results = {"knees forward error": None, "knees inward error": None}
    elif exercise_type == "OHP":
        results = {"elbows error": None, "knees error": None}

    error_details = error_parser(error_details_sentence, temporal=False)
    if error_details["errors"] is not None:
        for error in error_details["errors"]:
            if error is not None:
                results[f"{error} error"] = True

    return exercise_type, results


def extract_error_details_temporal(sentence):  # TEMPORAL

    # Split the sentence into individual error segments
    sentence = sentence.split(". ")
    sentence = [item for item in sentence if item != ""]

    assert len(sentence) == 2
    exercise_type_sentence = sentence[0].lstrip()
    error_details_sentence = sentence[1].lstrip()

    exercise_type_pattern = r"The subject is performing a ([A-Za-z]+)"
    exercise_type_match = re.search(exercise_type_pattern, exercise_type_sentence)
    exercise_type = exercise_type_match.group(1)

    results = {}
    if exercise_type == "Squat":
        results = {"knees forward error": None, "knees inward error": None}
    elif exercise_type == "OHP":
        results = {"elbows error": None, "knees error": None}

    error_details = error_parser(error_details_sentence, temporal=True)
    if error_details["errors"] is not None:
        for error in error_details["errors"]:
            if error is not None:
                # results[f"{error} error"] = True
                # The pattern to extract error_type, start_time, and end_time
                pattern = r"(?P<error_type>.+?) from frame time (?P<start_time>\d+\.\d+)s to (?P<end_time>\d+\.\d+)s"

                # Perform the match
                match = re.match(pattern, error)

                if match:
                    # Extract the variables using group names
                    error_type = match.group("error_type")
                    start_time = float(match.group("start_time"))
                    end_time = float(match.group("end_time"))
                    results[f"{error_type} error"] = (start_time, end_time)
                else:
                    raise Exception("Wrong sentence format in the output?")

    return exercise_type, results


# Parse the data and extract relevant information
def process_dataset(data):
    processed_data = {}

    for item in data:
        # Extract the ID, video path, and conversation details
        video_path = item.get("video")
        video_id = video_path.split("/")[-1].split(".")[0]
        conversations = item.get("conversations")

        # Find model predictions from the 'gpt' response
        # for conversation in conversations:
        #     if conversation.get("from") == "human":
        #         exercise_type = extract_exercise_type(conversation.get("value"))
        #     elif conversation.get("from") == "gpt":
        #         error_details = extract_error_details(conversation.get("value"))

        for conversation in conversations:
            if conversation.get("from") == "gpt":
                exercise_type, error_details = extract_error_details(conversation.get("value"))

        # Store structured data for each entry

        # check if it's a augmented video
        parent_folder = os.path.basename(os.path.dirname(video_path))
        augmentation_type_match = re.search(r"_(\w+)$", parent_folder)
        augmentation_type = augmentation_type_match.group(1) if augmentation_type_match else None

        if augmentation_type is not None:
            continue
            # video_id = f"{video_id}_{augmentation_type}"
        processed_data[video_id] = {"exercise_type": exercise_type, "video": video_path, "error_details": error_details}

    return processed_data


def model_inference(dataset, store=True):
    result_dataset = {}
    with torch.no_grad() and torch.cuda.amp.autocast():
        pretrained = "PATH_PRETRAINED_MODEL"
        base_model = None
        model_name = "llava_qwen"
        # model_name = "lora_llava_qwen"
        device = "cuda"
        device_map = "auto"

        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, base_model, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args

        model.eval()
        max_frames_num = 32

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        for id, value in tqdm(dataset.items()):
            gt_exercise_type = value["exercise_type"]
            video_path = os.path.join(dataset_path, value["video"])
            gt_error_details = value["error_details"]

            video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
            video = [video]

            if temporal_instructions:
                time_instruciton = (
                    f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                )
                question = f"{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\nThe subject is performing a Squat or a Overhead Press (OHP) exercise. Which one is he making? Is he making a mistake? If so, what mistake is he making?"
            else:
                question = f"{DEFAULT_IMAGE_TOKEN}\nThe subject is performing a Squat or a Overhead Press (OHP) exercise. Which one is he making? Is he making a mistake? If so, what mistake is he making?"

            if include_hpe_data:
                with open(get_keypoint_path(id, gt_exercise_type)) as f:
                    hpe_data = json.load(f)

                for data in hpe_data:
                    for data_instances in data["instances"]:
                        del data_instances["keypoint_scores"]
                input_message = f"{input_message} To help you identify the subject, these are the Human Pose Estimation data extract in a 3D space: {hpe_data}."

            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            cont = model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=True,
                num_beams=4,
                max_new_tokens=128,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            print(text_outputs)
            pr_exercise_type, pr_error_details = extract_error_details(text_outputs)
            result_dataset[id] = {
                "prediction": text_outputs,
                "prediction_exercise_type": pr_exercise_type,
                "prediction_error_details": pr_error_details,
                "ground_truth_exercise_type": gt_exercise_type,
                "ground_truth_error_details": gt_error_details,
                "video_path": video_path,
            }
    if store:
        temp_path = os.path.join(output_predictions_path, (pretrained.split("/")[-1]))
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        with open(os.path.join(temp_path, str(datetime.datetime.now().timestamp())) + ".json", "w") as file:
            json.dump(result_dataset, file, indent=4)
    return result_dataset


def load_prediction_from_json(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def get_predictions_from_fine_tuned_model(dataset, online=False):
    if online == True:
        return model_inference(dataset)
    else:
        filepath = "OUTPUT_INFERENCE.json"
        return load_prediction_from_json(filepath)


online = True
include_hpe_data = False
temporal_data = True
temporal_instructions = temporal_data


if temporal_data == True:
    extract_error_details = extract_error_details_temporal
else:
    extract_error_details = extract_error_details_boolean

filepath = "uamp25_87_files/build_llava_dataset/test.json"  # Test Set
data = load_json_file(filepath)
test_set = process_dataset(data)


predictions = get_predictions_from_fine_tuned_model(test_set, online=online)

# Evaluate the test set
predicted_label = {}
ground_truth_label = {}

for id, value in predictions.items():
    if id in test_set.keys():
        # ignore augmented videos
        # augmentation_type = id.split("_")
        # if len(augmentation_type) > 2:
        #     augmentation_type = augmentation_type[-1]
        # else:
        #     augmentation_type = None
        # if augmentation_type is not None:
        #     continue

        prediction = value["prediction_error_details"]
        if type(prediction) != dict:
            _, prediction = extract_error_details(prediction)
        ground_truth = value["ground_truth_error_details"]
        if type(ground_truth) != dict:
            _, ground_truth = extract_error_details(ground_truth)

        # compare the pair of predictions and ground truth
        ground_truth_temp = test_set[id]["error_details"]
        if ground_truth != ground_truth_temp:
            print(f"Error in preprocessing: {id} has different ground truth error details.")
        ground_truth = ground_truth_temp
        if len(ground_truth) != len(prediction):
            _, prediction = extract_error_details(value["prediction"])

        assert len(ground_truth) == len(prediction)

        if set(ground_truth.keys()) != set(prediction.keys()):
            print("wrong exercise type predicted")
            continue

        for key in ground_truth.keys():

            gt_present = 0  # ground_truth["present"]
            gt_value = ground_truth[key]
            if ground_truth[key] != None:
                gt_present = 1
            pred_present = 0  # value["prediction"][error_type]["present"]
            pred_value = prediction[key]
            if prediction[key] != None:
                pred_present = 1

            if key not in ground_truth_label:
                ground_truth_label[key] = {"present": [], "value": []}
                predicted_label[key] = {"present": [], "value": []}

            # ground_truth_label[key].append(gt_present)
            # predicted_label[key].append(pred_present)
            ground_truth_label[key]["present"].append(gt_present)
            ground_truth_label[key]["value"].append(gt_value)
            predicted_label[key]["present"].append(pred_present)
            predicted_label[key]["value"].append(pred_value)
    else:
        print(f"Error in the dataset: {id} not found in the test set.")

# Aggregate metrics
precision_original, recall_original, f1_original, accuracy_original, ap_original = {}, {}, {}, {}, {}
for key in ground_truth_label.keys():
    precision_original[key] = precision_score(y_true=ground_truth_label[key]["present"], y_pred=predicted_label[key]["present"])
    recall_original[key] = recall_score(y_true=ground_truth_label[key]["present"], y_pred=predicted_label[key]["present"])
    f1_original[key] = f1_score(y_true=ground_truth_label[key]["present"], y_pred=predicted_label[key]["present"])
    accuracy_original[key] = accuracy_score(y_true=ground_truth_label[key]["present"], y_pred=predicted_label[key]["present"])

    # TEMPORAL metric
    if temporal_data == True:
        ap_original[key] = compute_ap_score(y_true=ground_truth_label[key]["value"], y_pred=predicted_label[key]["value"])
if temporal_data == True:
    map_original = np.mean(list(ap_original.values()))

print("Binary Classification Metrics Original Dataset:")
print(f"Accuracy: {accuracy_original}")
print(f"Precision: {precision_original}")
print(f"Recall: {recall_original}")
print(f"F1: {f1_original}")
if temporal_data == True:
    print(f"AP: {ap_original}")
    print(f"mAP: {map_original}")
