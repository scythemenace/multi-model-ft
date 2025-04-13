import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np

# Step 1: Load the CSV files
train_data = pd.read_csv("input_data/TrainData.csv", header=None, names=["text"])
train_labels = pd.read_csv("input_data/TrainLabels.csv", header=None, names=["label"])
validation_data = pd.read_csv("input_data/ValidationData.csv", header=None, names=["text"])
validation_labels = pd.read_csv("input_data/ValidationLabels.csv", header=None, names=["label"])
test_data = pd.read_csv("input_data/TestData.csv", header=None, names=["text"])

# Determine the number of unique labels
unique_labels = set(train_labels["label"]).union(set(validation_labels["label"]))
num_labels = len(unique_labels)

# Map labels to [0, num_labels - 1] if necessary
label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
train_labels["label"] = train_labels["label"].map(label_map)
validation_labels["label"] = validation_labels["label"].map(label_map)

# Combine data and labels into a single DataFrame for train and validation
train_df = train_data.copy()
train_df["labels"] = train_labels["label"]
validation_df = validation_data.copy()
validation_df["labels"] = validation_labels["label"]
test_df = test_data.copy()  # Test data has no labels

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

# Create a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize the datasets
tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_validation = dataset["validation"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)

# Ensure labels are integers (if not already)
def convert_labels(examples):
    examples["labels"] = [int(label) for label in examples["labels"]]
    return examples

tokenized_train = tokenized_train.map(convert_labels, batched=True)
tokenized_validation = tokenized_validation.map(convert_labels, batched=True)
# No labels for test data, so skip convert_labels for test

# Remove unnecessary columns and set format
columns_to_keep = ["input_ids", "attention_mask", "labels"]
tokenized_train = tokenized_train.remove_columns(
    [col for col in tokenized_train.column_names if col not in columns_to_keep]
)
tokenized_validation = tokenized_validation.remove_columns(
    [col for col in tokenized_validation.column_names if col not in columns_to_keep]
)
# Test data won't have labels
tokenized_test = tokenized_test.remove_columns(
    [col for col in tokenized_test.column_names if col not in ["input_ids", "attention_mask"]]
)

tokenized_train.set_format("torch", columns=columns_to_keep)
tokenized_validation.set_format("torch", columns=columns_to_keep)
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask"])

# Load the model with the correct number of labels
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels, torch_dtype="auto"
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="news_article_classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5, # tweaked
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5, # tweaked
    weight_decay=0.01,   # tweaked
    metric_for_best_model="accuracy",
)

# Load the accuracy metric
metric = evaluate.load("accuracy")

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Optional: Generate predictions on test data (no labels, so no evaluation)
test_predictions = trainer.predict(tokenized_test)
test_logits = test_predictions.predictions
test_pred_labels = np.argmax(test_logits, axis=-1)

# Save test predictions to a CSV
test_df["predicted_label"] = test_pred_labels
test_df.to_csv("input_data/TestData_with_predictions.csv", index=False)
print("Test predictions saved to TestData_with_predictions.csv")