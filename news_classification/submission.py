import pandas as pd

submissionType = 1  # This must be included and is used to tell our code that this is an integer array submission

def print_predictions():
    # Load the test predictions from the CSV
    test_predictions_df = pd.read_csv("input_data/TestData_with_predictions.csv")
    
    # Extract the predicted labels as a list
    predicted_labels = test_predictions_df["predicted_label"].tolist()
    
    # Print the predictions in a copy-pasteable format
    print("[", end="")
    for i, label in enumerate(predicted_labels):
        if i < len(predicted_labels) - 1:
            print(f"{label}, ", end="")
        else:
            print(f"{label}]")

# Run the function to print the predictions
print_predictions()