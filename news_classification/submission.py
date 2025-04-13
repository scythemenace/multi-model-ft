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

def predictionResults():
    
    return [0, 2, 2, 3, 0, 0, 4, 2, 0, 3, 3, 4, 0, 0, 2, 3, 0, 4, 2, 0, 2, 3, 0, 0, 0, 3, 0, 0, 0, 7, 3, 3, 0, 0, 2, 2, 3, 3, 0, 3, 3, 2, 0, 4, 0, 3, 3, 0, 0, 2, 0, 0, 2, 3, 3, 0, 0, 3, 0, 2, 2, 2, 2, 2, 2, 0, 0, 3, 0, 3, 3, 2, 0, 3, 7, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 7, 0, 4, 7, 6, 2, 0, 4, 0, 0, 0, 0, 4, 2, 0, 3, 3, 3, 0, 0, 2, 2, 7, 0, 2, 7, 0, 2, 0]