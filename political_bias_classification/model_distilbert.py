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
from sklearn.model_selection import train_test_split

# Step 1: Load the CSV files
train_df = pd.read_csv("input_data/training.csv")
validation_df = pd.read_csv("input_data/validation.csv", header=None, names=["text"])
test_df = pd.read_csv("input_data/tweets_test.csv", header=None, names=["text"])

# Step 2: Split training data into train and validation sets (since validation.csv has no labels)
train_df = train_df.rename(columns={"tweet": "text", "label": "labels"})  # Rename columns to match expected format
train_split, val_split = train_test_split(train_df, test_size=0.2, stratify=train_df["labels"], random_state=42)

# Step 3: Determine the number of unique labels
unique_labels = set(train_df["labels"])
num_labels = len(unique_labels)  # Should be 5 (labels 1 to 5)

# Map labels to [0, num_labels - 1] (1 to 5 → 0 to 4)
label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
train_split["labels"] = train_split["labels"].map(label_map)
val_split["labels"] = val_split["labels"].map(label_map)

# Add placeholder labels for validation and test sets (not used for training/evaluation)
validation_df["labels"] = 0
test_df["labels"] = 0

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_split[["text", "labels"]])
validation_dataset = Dataset.from_pandas(val_split[["text", "labels"]])  # Use split validation for evaluation
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
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
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

# Step 13: Evaluate on the split validation set
eval_results = trainer.evaluate()
print(f"Validation Accuracy (on split validation set): {eval_results['eval_accuracy']:.4f}")

# Step 14: Generate predictions on test data
test_predictions = trainer.predict(tokenized_test)
test_logits = test_predictions.predictions
test_pred_labels = np.argmax(test_logits, axis=-1)

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