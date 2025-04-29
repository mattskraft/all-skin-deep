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
- [Web Demo](#web-app-demo)
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
│   │   ├── prepare_dataset_stage_1.py
│   │   ├── prepare_dataset_stage_2.py
│   ├── nst/
│   │   ├── nst_batch.py
│   │   ├── nst_single.py
│   │   ├── nst_utils.py
│   ├── training/
│   │   ├── training_stage_1.py
│   │   ├── training_stage_2.py
│   │   ├── training_utils.py
│   ├── config.py
└── outputs/
    ├── plots/
    └── models/         # training history, trained models
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
python prepare_dataset_stage_1.py \
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
python nst_batch.py \
  --content data/processed/dataset_balanced \
  --style data/processed/style-images-resized \
  --output data/processed/dataset_balanced_nst
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
python prepare_dataset_stage_2.py \
  --original_dir data/processed/dataset_balanced \
  --st_dir data/processed/dataset_balanced_nst \
  --output_dir data/processed/dataset_stage2
```

---

### Model Training

Model training occurs in two stages:

#### Stage 1
In the first stage, a base model is fine-tuned on the unbalanced stage 1 dataset. The base model is a pre-trained MobileNetv2 with all but the last layer block frozen and a classification head on top. Macro F1 Score is used as validation metric.

```bash
python training_stage_1.py \
  --train_dir data/processed/dataset_stage1/train \
  --val_dir data/processed/dataset_stage1/val
```
#### Stage 2
In the second stage, the best model from stage 1 is further fine-tuned, now on the NST-augmented, balanced stage 2 dataset, using a cross-style approach.

```bash
python training_stage_2.py \
  --data_dir data/processed/dataset_stage2 \
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

## Web APP Demo

A live interactive demo of the final model is available:

👉 [Try the Skin Lesion Classifier in your browser](https://precious-haupia-39d814.netlify.app/)

The model was converted to TensorFlowJS format and deployed as a static web app.  
This allows real-time skin lesion predictions directly in the browser without needing a server.

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

## Notes

- GPU strongly recommended for NST image generation
- Random seeds are set for reproducibility

## Acknowledgments

This project was inspired and guided by the following works:

- [Skin Lesion Analyzer + TensorFlow.js Web App](https://www.kaggle.com/code/vbookshelf/skin-lesion-analyzer-tensorflow-js-web-app)  
  Motivation for using a lightweight CNN architecture (MobileNetV2) with a focus on data privacy

- [Improving Skin Color Diversity in Cancer Detection: Deep Learning Approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC10334920/)  
  Inspiration for applying Neural Style Transfer (NST) for dataset augmentation, and for evaluating image quality using SSIM and BRISQUE

## License

MIT License

