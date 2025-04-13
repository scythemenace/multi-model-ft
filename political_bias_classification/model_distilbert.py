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
train_df = pd.read_csv("input_data/training.csv")
validation_df = pd.read_csv("input_data/validation.csv")
# Load test data with header=0 to treat the first row as a header, and select only the "tweet" column
test_df = pd.read_csv("input_data/tweets_test.csv", header=0, usecols=["tweet"])

# Debug: Check the number of rows in each dataset
print(f"Number of rows in train_df: {len(train_df)}")
print(f"Number of rows in validation_df: {len(validation_df)}")
print(f"Number of rows in test_df: {len(test_df)}")

# Step 2: Prepare the data by renaming columns for consistency
train_df = train_df.rename(columns={"tweet": "text", "label": "labels"})
validation_df = validation_df.rename(columns={"tweet": "text", "label": "labels"})
test_df = test_df.rename(columns={"tweet": "text"})

# Step 3: Determine the number of unique labels
unique_labels = set(train_df["labels"]).union(set(validation_df["labels"]))
num_labels = len(unique_labels)  # Should be 5 (labels 1 to 5)

# Map labels to [0, num_labels - 1] (1 to 5 → 0 to 4)
label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
train_df["labels"] = train_df["labels"].map(label_map)
validation_df["labels"] = validation_df["labels"].map(label_map)

# Add placeholder labels for test set (not used for training/evaluation)
test_df["labels"] = 0

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df[["text", "labels"]])
validation_dataset = Dataset.from_pandas(validation_df[["text", "labels"]])
test_dataset = Dataset.from_pandas(test_df[["text", "labels"]])

# Create a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

# Step 4: Load the tokenizer and model
model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, torch_dtype="auto"
)

# Step 5: Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Tokenize the datasets
tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_validation = dataset["validation"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)

# Step 6: Ensure labels are integers
def convert_labels(examples):
    examples["labels"] = [int(label) for label in examples["labels"]]
    return examples

tokenized_train = tokenized_train.map(convert_labels, batched=True)
tokenized_validation = tokenized_validation.map(convert_labels, batched=True)
# No labels for test data, so skip convert_labels for test

# Step 7: Remove unnecessary columns and set format
columns_to_keep = ["input_ids", "attention_mask", "labels"]
tokenized_train = tokenized_train.remove_columns(
    [col for col in tokenized_train.column_names if col not in columns_to_keep]
)
tokenized_validation = tokenized_validation.remove_columns(
    [col for col in tokenized_validation.column_names if col not in columns_to_keep]
)
tokenized_test = tokenized_test.remove_columns(
    [col for col in tokenized_test.column_names if col not in ["input_ids", "attention_mask"]]
)

tokenized_train.set_format("torch", columns=columns_to_keep)
tokenized_validation.set_format("torch", columns=columns_to_keep)
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask"])

# Step 8: Define training arguments
training_args = TrainingArguments(
    output_dir="tweet_bias_classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=4e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=7,
    weight_decay=0.1,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
)

# Step 9: Load the accuracy metric
metric = evaluate.load("accuracy")

# Step 10: Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Step 11: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    compute_metrics=compute_metrics,
)

# Step 12: Start training
trainer.train()

# Step 13: Evaluate on the validation set
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

# Step 14: Generate predictions on test data
test_predictions = trainer.predict(tokenized_test)
test_logits = test_predictions.predictions
test_pred_labels = np.argmax(test_logits, axis=-1)

# Verify the number of predictions
print(f"Number of test predictions: {len(test_pred_labels)}")

# Step 15: Create inverse label map to convert model labels (0-4) back to original labels (1-5)
inverse_label_map = {new_label: old_label for old_label, new_label in label_map.items()}

# Map predictions back to original label range (1-5)
test_pred_labels_mapped = [inverse_label_map[label] for label in test_pred_labels]

# Step 16: Save test predictions to a CSV with only one column named "Label"
output_df = pd.DataFrame(test_pred_labels_mapped, columns=["Label"])
output_df.to_csv("input_data/cleaned_combined_result.csv", index=False)
print("Test predictions saved to cleaned_combined_result.csv")

# Step 17: Print test predictions for submission
print("Predictions for submission (test set):")
print("[", end="")
for i, label in enumerate(test_pred_labels_mapped):
    if i < len(test_pred_labels_mapped) - 1:
        print(f"{label}, ", end="")
    else:
        print(f"{label}]")