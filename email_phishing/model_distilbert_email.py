import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import evaluate
import zipfile

# Step 1: Load the CSV files
def load_data(file_path, has_labels=True):
    if has_labels:
        # Training/validation: col0=email, col1=text, col2=label
        df = pd.read_csv(file_path, header=None, usecols=[1, 2], names=["text", "labels"])
    else:
        # Testing: col0=email, col1=text
        df = pd.read_csv(file_path, header=None, usecols=[1], names=["text"])
    return df

train_df = load_data("input_data/training.csv", has_labels=True)
validation_df = load_data("input_data/validation.csv", has_labels=True)
test_df = load_data("input_data/testing.csv", has_labels=False)

# Step 2: Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

# Create a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

# Step 3: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize the datasets
tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_validation = dataset["validation"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)

# Ensure labels are integers
def convert_labels(examples):
    examples["labels"] = [int(label) for label in examples["labels"]]
    return examples

tokenized_train = tokenized_train.map(convert_labels, batched=True)
tokenized_validation = tokenized_validation.map(convert_labels, batched=True)
# No labels for test data

# Remove unnecessary columns and set format
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

# Step 4: Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, torch_dtype="auto"
)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="spam_classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
)

# Step 6: Load metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

# Step 7: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    compute_metrics=compute_metrics,
)

# Step 8: Train the model
trainer.train()

# Step 9: Generate predictions on test data
test_predictions = trainer.predict(tokenized_test)
test_logits = test_predictions.predictions
test_pred_labels = np.argmax(test_logits, axis=-1)

# Step 10: Save predictions to answer.txt
with open("answer.txt", "w") as f:
    for label in test_pred_labels:
        f.write(f"{label}\n")