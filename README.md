<a href="pdf/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-blue"/></a>
<a href="pdf/Poster.pdf"><img src="https://img.shields.io/badge/Poster-PDF-red"/></a>

# Fine-Tuning Large Multimodal Models for Fitness Action Quality Assessment

## Usage

The dataset [Fitness-AQA](https://github.com/ParitoshParmar/Fitness-AQA?tab=readme-ov-file) should be placed in the `archives_data` folder.

- All zip files should be extracted in the same location where they are.
- For more details, compare your file tree with ours: [before](archives_data/files_before_preprocessing.txt) / [after](archives_data/files_after_preprocessing.txt) — preprocessing.
- Build the dataset in conversation style using the [build_llava_dataset_dynamic_step.py](uamp25_87_files/build_llava_dataset/build_llava_dataset_dynamic_step.py) or [build_llava_dataset_two_step](uamp25_87_files/build_llava_dataset/build_llava_dataset_two_step.py) script.

### To train a model, you should:

- Build the Singularity container using the definition file in [`requirements`](requirements.txt).
- We modified the original [training script](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/video/train/SO400M_Qwen2_7B_ov_to_video_am9.sh). Our version of the script is the [`SO400M_Qwen2_7B_ov_to_video_am9.sh`](SO400M_Qwen2_7B_ov_to_video_am9.sh) script.

- **Note**: We used 4 GPUs, each with 64GB of memory (A100s), for training.

### To evaluate a model, you should:

- Run the [`evaluation.py`](evaluation.py).
