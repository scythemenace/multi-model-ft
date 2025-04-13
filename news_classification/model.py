submissionType = 0  # This must be included and is used to tell our code that this is a model submission

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
import torch
from collections import Counter

class Model:
    def __init__(self):
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.num_labels = None  # Will be set in fit
        self.model = None  # Will be initialized in fit
        self.trainer = None  # Will be set in fit
        self.class_weights = None  # Will be set in fit

    def fit(self, X_train, y_train):
        """
        This should handle the logic of training your model
        :param X_train: 1 dimensional np.array of np.arrays containing one string entry each corresponding to the training point
        :param y_train: 1 dimensional np.array of the same length as X_train containing ground truth labels as integers
        """
        # Convert X_train and y_train to a DataFrame
        train_df = pd.DataFrame({
            "text": [x[0] for x in X_train],  # Extract the string from each inner array
            "labels": y_train
        })

        # Determine the number of unique labels
        unique_labels = set(train_df["labels"])
        self.num_labels = len(unique_labels)  # Should be 8 for your dataset
        # Ensure labels are in [0, num_labels - 1]
        label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
        train_df["labels"] = train_df["labels"].map(label_map)

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_df)

        # Split into train/validation (80/20 split)
        dataset = DatasetDict({"train": train_dataset})
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        tokenized_train = dataset["train"]
        tokenized_validation = dataset["test"]

        # Tokenize the datasets
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

        tokenized_train = tokenized_train.map(tokenize_function, batched=True)
        tokenized_validation = tokenized_validation.map(tokenize_function, batched=True)

        # Ensure labels are integers
        def convert_labels(examples):
            examples["labels"] = [int(label) for label in examples["labels"]]
            return examples

        tokenized_train = tokenized_train.map(convert_labels, batched=True)
        tokenized_validation = tokenized_validation.map(convert_labels, batched=True)

        # Remove unnecessary columns and set format
        columns_to_keep = ["input_ids", "attention_mask", "labels"]
        tokenized_train = tokenized_train.remove_columns(
            [col for col in tokenized_train.column_names if col not in columns_to_keep]
        )
        tokenized_validation = tokenized_validation.remove_columns(
            [col for col in tokenized_validation.column_names if col not in columns_to_keep]
        )

        tokenized_train.set_format("torch", columns=columns_to_keep)
        tokenized_validation.set_format("torch", columns=columns_to_keep)

        # Compute class weights for imbalanced data
        label_counts = Counter(train_df["labels"])
        total_samples = len(train_df)
        self.class_weights = torch.tensor([total_samples / (self.num_labels * label_counts[i]) for i in range(self.num_labels)]).to("cuda")

        # Load the model with dropout
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=self.num_labels,
            torch_dtype="auto",
            classifier_dropout=0.3
        )

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Custom Trainer to use class weights in the loss function
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="news_article_classifier_revised",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,  # Reduced from 7 to prevent overfitting
            weight_decay=0.2,  # Increased from 0.1 for stronger regularization
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        # Load the accuracy metric
        metric = evaluate.load("accuracy")

        # Define the compute_metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        # Initialize the custom Trainer
        self.trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_validation,
            compute_metrics=compute_metrics,
        )

        # Start training
        self.trainer.train()

    def predict(self, X_test):
        """
        This should handle making predictions with a trained model
        :param X_test: 1 dimensional np.array of np.arrays containing one string entry each corresponding to the testing point
        :return: 1 dimensional np.array of the same length as X_test containing predictions to each point in X_test as integers
        """
        # Convert X_test to a DataFrame
        test_df = pd.DataFrame({
            "text": [x[0] for x in X_test]  # Extract the string from each inner array
        })

        # Convert to Hugging Face Dataset
        test_dataset = Dataset.from_pandas(test_df)

        # Tokenize the test data
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

        tokenized_test = test_dataset.map(tokenize_function, batched=True)

        # Remove unnecessary columns and set format
        tokenized_test = tokenized_test.remove_columns(
            [col for col in tokenized_test.column_names if col not in ["input_ids", "attention_mask"]]
        )
        tokenized_test.set_format("torch", columns=["input_ids", "attention_mask"])

        # Generate predictions
        test_predictions = self.trainer.predict(tokenized_test)
        test_logits = test_predictions.predictions
        test_pred_labels = np.argmax(test_logits, axis=-1)

        # Convert predictions to a 1D NumPy array of integers
        return test_pred_labels.astype(np.int32)