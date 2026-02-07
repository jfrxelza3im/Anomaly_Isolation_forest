import pandas as pd
import joblib
from processing_Piplan import
 #test data churuk of the dataset
test_batch = pd.read_csv(r"Data_Processed/test_data.csv")
new_data = pd.read_csv(r"Data_Processed/RealTime_sample.csv")

# function for batch inference
def predict_Batch(test_batch):

    model = joblib.load("isolation_forest_model.joblib")

    # Make predictions using the loaded model
    predictions = model.predict(test_batch)

    return predictions
# Example usage
if __name__ == "__main__":
    # Load the test batch (replace with actual file path)
    test_batch = pd.read_csv("Data_Processed/test_data.csv")

    # Get predictions for the test batch
    predictions = predict_Batch(test_batch)

    # Print the predictions
    print(predictions)

def predict_real_time(new_data):

    model = joblib.load("isolation_forest_model.joblib")
    prediction = model.predict([new_data])

    return prediction