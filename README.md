# Text Classification Tasks

This repository contains the implementation of three text classification tasks for different datasets, using DistilBERT models for classification. The tasks are:

- **News Article Classification**: Classifies news articles into 8 categories.
- **Political Bias Tweet Classification**: Classifies tweets into 5 political bias categories.
- **Phishing Email Classification**: Classifies emails as spam (0) or non-spam (1).

The report for all tasks is included as `report.pdf`.

## Directory Structure

- **news_classification/**
  - `input_data/`: Contains input CSV files for the task.
  - `model_news_classification.py`: Script to train and evaluate the DistilBERT model.
  - `model.py`: Additional model utilities (if needed).
  - `submission.py`: Script to generate predictions for submission.
- **political_bias_classification/**
  - `input_data/`: Contains input CSV files for the task.
  - `distilbert.zip`: Zipped submission file for Codabench.
  - `model_distilbert.py`: Script to train and evaluate the base DistilBERT model.
  - `model_fine_tuned.py`: Script for running a fine-tuned DistilBERT model for a similar task.
- **email_phishing/**
  - `input_data/`: Contains input CSV files for the task.
  - `answer.txt`: Test predictions for Codabench submission.
  - `answer.txt.zip`: Zipped submission file for Codabench.
  - `model_distilbert_email.py`: Script to train and evaluate the DistilBERT model.
- `report.pdf`: Report detailing the approaches and results for all tasks.
- `requirements.txt`: List of required Python packages.

## Setup Instructions

1. **Install Python**: Ensure you have Python 3.8 or higher installed on your system.
2. **Install Dependencies**: Use the provided `requirements.txt` to install necessary packages. Run the following command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```
   This will install libraries like `transformers`, `pandas`, `torch`, `sklearn`, and `evaluate`.

## Running the Scripts

Each task has a dedicated script to train the model and generate predictions. Navigate to the respective directory and run the script using Python 3.

- **News Article Classification**:

  ```bash
  cd news_classification
  python3 model_news_classification.py
  ```

- **Political Bias Tweet Classification**:

  ```bash
  cd political_bias_classification
  python3 model_distilbert.py
  ```

  Alternatively, to use the fine-tuned model script:

  ```bash
  python3 model_fine_tuned.py
  ```

- **Phishing Email Classification**:
  ```bash
  cd email_phishing
  python3 model_distilbert_email.py
  ```

Each script will train the model, evaluate it on the validation set, and generate test predictions, saving them in the required format for Codabench submission (e.g., `answer.txt` for Phishing Email Classification) or you can manually create zip files of the output csv files based on the instructions of the respective codebench pages.

## Computational Resources

This project was executed using high compute resources provided by McMaster University, specifically an Nvidia 4090 GPU. Ensure you have adequate computational resources (e.g., a GPU with at least 12GB VRAM) to run the scripts efficiently. If local resources are insufficient, consider using cloud platforms like Google Colab, AWS, or Azure, which offer GPU support.
