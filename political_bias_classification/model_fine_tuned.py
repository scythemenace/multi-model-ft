import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset

# Step 1: Load the data
train_df = pd.read_csv("input_data/training.csv")
validation_df = pd.read_csv("input_data/validation.csv", header=None, names=["tweet"])
test_df = pd.read_csv("input_data/tweets_test.csv", header=None, names=["tweet"])

# Step 2: Split training data into train and validation sets (since validation.csv has no labels)
train_split, val_split = train_test_split(train_df, test_size=0.2, stratify=train_df["label"], random_state=42)

# Step 3: Load the pretrained model and tokenizer using Auto classes
model_name = "bucketresearch/politicalBiasBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, ignore_mismatched_sizes=True)

# Step 4: Preprocess the data
def preprocess_function(examples):
    return tokenizer(examples["tweet"], truncation=True, padding="max_length", max_length=128)

# Convert pandas DataFrames to Hugging Face Datasets
train_split["label"] = train_split["label"].astype(int) - 1  # Shift labels to 0-4
val_split["label"] = val_split["label"].astype(int) - 1
validation_df["label"] = 0  # Placeholder for validation set (no true labels)
test_df["label"] = 0  # Placeholder for test set

train_dataset = Dataset.from_pandas(train_split[["tweet", "label"]])
val_dataset = Dataset.from_pandas(val_split[["tweet", "label"]])
validation_dataset = Dataset.from_pandas(validation_df[["tweet", "label"]])
test_dataset = Dataset.from_pandas(test_df[["tweet", "label"]])

# Tokenize the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
validation_dataset = validation_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
validation_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Step 5: Define compute_metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

# Step 6: Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Step 7: Initialize and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Use the split validation set for evaluation
    compute_metrics=compute_metrics,
)

trainer.train()

# Step 8: Evaluate on the split validation set
eval_results = trainer.evaluate()
print(f"Validation Accuracy (on split validation set): {eval_results['eval_accuracy']:.4f}")

# Get validation predictions for detailed report (on split validation set)
val_predictions = trainer.predict(val_dataset)
y_pred_val = val_predictions.predictions.argmax(-1)
y_val = val_predictions.label_ids
print("Classification Report (on split validation set):")
print(classification_report(y_val, y_pred_val))

# Step 9: Predict on the actual validation set (for submission, no evaluation possible)
validation_predictions = trainer.predict(validation_dataset)
y_pred_validation = validation_predictions.predictions.argmax(-1)
y_pred_validation = y_pred_validation + 1  # Shift back to 1-5 for submission

# Step 10: Predict on test set
test_predictions = trainer.predict(test_dataset)
y_pred_test = test_predictions.predictions.argmax(-1)
y_pred_test = y_pred_test + 1  # Shift back to 1-5 for submission

# Save predictions
test_df["predicted_label"] = y_pred_test
test_df.to_csv("input_data/TestData_with_predictions_politicalBiasBERT.csv", index=False)
print("Test predictions saved to TestData_with_predictions_politicalBiasBERT.csv")

# Print test predictions for submission
print("Predictions for submission (test set):")
print("[", end="")
for i, label in enumerate(y_pred_test):
    if i < len(y_pred_test) - 1:
        print(f"{label}, ", end="")
    else:
        print(f"{label}]")