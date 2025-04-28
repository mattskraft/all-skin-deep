# Skin Lesion Classification with Neural Style Transfer Augmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Active-brightgreen)

This repository contains the code and workflows for preparing a skin lesion dataset (HAM10000), augmenting it using Neural Style Transfer (NST), and training a deep learning model for skin lesion classification.

## Table of Contents

- [Project Structure](#project-structure)
- [Workflow](#workflow)
  - [Download and Prepare Raw Data](#download-and-prepare-raw-data)
  - [Clean and Explore the Dataset](#clean-and-explore-the-dataset)
  - [Prepare Initial Datasets](#prepare-initial-datasets)
  - [Generate Style-Transferred Images](#generate-style-transferred-images)
  - [Explore NST Images](#explore-nst-images)
  - [Prepare Final Combined Dataset](#prepare-final-combined-dataset)
  - [Fine-Tune the Model](#fine-tune-the-model)
  - [Evaluate Model Performance](#evaluate-model-performance)
- [Requirements](#requirements)
- [Notes](#notes)
- [License](#license)

## Project Structure

```plaintext
/
├── README.md
├── requirements.txt
├── data/                # all datasets
│   ├── raw/
│   ├── processed/
├── notebooks/           # Jupyter notebooks
├── scripts/             # project code
│   ├── prepare_datasets/
│   │   ├── prepare_datasets_1_and_2.py
│   │   ├── prepare_dataset_3.py
│   ├── nst/
│   │   ├── nst_main.py
│   │   ├── nst_utils.py
│   ├── training/
│   │   ├── finetune_orig.py
│   │   ├── finetune_cross.py
│   │   ├── model_utils.py
│   ├── config.py
└── outputs/             # trained models, evaluation results
    ├── plots/
    └── models/
```

## Workflow

### Download and Prepare Raw Data

- Download the **HAM10000** dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Unpack both ZIP files and place all images in:
  ```
  data/raw/HAM10000/images
  ```
- Place the metadata CSV file in:
  ```
  data/raw/HAM10000/
  ```

---

### Clean and Explore the Dataset

Run the notebook:

```
notebooks/01_data_exploration_and_cleaning.ipynb
```

- Explore the dataset and become aware of the gross class imbalance
- Remove duplicate images and ave a cleaned metadata CSV (`HAM10000_metadata_clean.csv`)

---

### Prepare Initial Datasets

Prepare datasets for training:

```bash
python scripts/prepare_datasets/prepare_datasets_1_and_2.py \
  --input_dir data/raw/HAM10000/images \
  --metadata data/raw/HAM10000/HAM10000_metadata_clean.csv \
  --output_dir data/processed/ \
  --random_state 42
```

Creates:

- A **test dataset** that is held out completely, both during neural style transfer and during model fine-tuning
- An **unbalanced dataset** (with train/validation split) that becomes the training data for the stage 1 of fine-tuning
- A **balanced dataset** with all images from the smallest class and a random selection of the same number of images from the larger classes

---

### Generate Style-Transferred Images

Apply Neural Style Transfer to augment images:

```bash
python scripts/nst/nst_main.py \
  --content data/processed/dataset_balanced \
  --style data/processed/style-images-resized \
  --output data/processed/nst_output
```

Generates style-transferred versions of the balanced dataset. Uses the provided [style images](data/processed/style-images-resized/).

For detailed information about NST processing, see [NST README](scripts/nst/README.md).

---

### Explore NST Images

Run the notebook:

```
notebooks/02_examine_and_select_nst_images.ipynb
```

- Explore and examine generated NST images
- Run quantitative evaluations and remove images that do not meet requirements

---

### Prepare Stage 2 Dataset

Combine the original balanced and the NST dataset to create the stage 2 dataset for fine-tuning:

```bash
python scripts/prepare_datasets/prepare_dataset_3.py \
  --original_dir data/processed/dataset_balanced \
  --st_dir data/processed/nst_output \
  --output_dir data/processed/dataset_combined
```

---

### Model Training

Model training occurs in two stages:

#### Stage 1
In the first stage, a base model is fine-tuned on the unbalanced stage 1 dataset. The base model is a pre-trained MobileNetv2 with all but the last layer block frozen and a classification head on top. Given the clinical setting, macro F1 Score is used as validation metric.

```bash
python scripts/training/finetune_orig.py \
  --train-dir data/processed/dataset_stage1/train \
  --val-dir data/processed/dataset_stage1/val
```
#### Stage 1
In the second stage, the best model from stage 1 is further fine-tuned, now on the NST-augmented, balanced stage 2 dataset.

```bash
python scripts/training/finetune_cross.py \
  --train-dir data/processed/dataset_stage2/train \
  --val-dir data/processed/dataset_stage2/val \
  --base_model outputs/models/finetune_orig_best.h5
```

---

### Evaluate Model Performance

Run the notebook:

```
notebooks/03_eval_model_on_testset.ipynb
```

- Assess the stage 1 and stage 2 trainings by ploting the training history of each
- Evaluate both models by running inference on each using the test set and plotting  some metrics:
    - Confusion matrix
    - Class-wise and macro F1 scores
    - Class-wise and top-3 classification accuracies

---

## Requirements

- Python 3.8+
- PyTorch
- TensorFlow / Keras
- scikit-learn
- pandas, numpy, matplotlib, tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Notes

- GPU strongly recommended for NST image generation
- Random seeds are set for reproducibility

---

## License

MIT License

