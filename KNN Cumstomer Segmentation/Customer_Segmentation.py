import pickle
import numpy as np
import warnings
import os
from dotenv import load_dotenv 



warnings.filterwarnings("ignore")

class CustomerSegmentor:
    """
    A class to assign a customer to a segment using a pre-trained KMeans model and scaler.
    """
    model = None
    scaler = None

    @staticmethod
    def load_pickle(path):
        """Load an object from a pickle file."""
        with open(path, 'rb') as file:
            return pickle.load(file)

    @classmethod
    def initialize(cls, model_path, scaler_path):
        """Initialize the class with a KMeans model and a scaler."""
        cls.model = cls.load_pickle(model_path)
        cls.scaler = cls.load_pickle(scaler_path)

    @staticmethod
    def get_input():
        """Get user input for clustering."""
        Age = int(input("Enter Age: "))
        Gender = float(input("Enter 0 for Male, 1 for Female: "))
        Annual_income = float(input("Enter Annual income (in k$): "))
        Spending_score = int(input("Enter Spending Score (1–100): "))
        return Age, Annual_income, Spending_score, Gender

    @classmethod
    def predict(cls, input_tuple):
        """Predict the customer segment for the input features."""
        input_array = np.array(input_tuple).reshape(1, -1)
        input_scaled = cls.scaler.transform(input_array)
        prediction = cls.model.predict(input_scaled)
        
    
        labels = [
            "Group 0",
            "Group 1",
            "Group 2",
            "Group 3",
            "Group 4",
            "Group 5"
        ]
        
        
        descriptive_labels = [
            "Group 1 - Target Customers (High Income, High Spending)",
            "Group 2 - Careful Spenders (High Income, Low Spending)",
            "Group 3 - Standard Customers (Average Income, Average Spending)",
            "Group 4 - Frugal Customers (Low Income, Low Spending)",
            "Group 5 - Careless Spenders - Young People(Low Income, High Spending)",
            "Group 6  - Seniors (Older, Moderate Income, Moderate Spending)"
        ]

        
        return descriptive_labels[prediction[0]]


# ===== Main Code =====
if __name__ == "__main__":
    # Get paths from environment variables loaded from the .env file
    # Load variables from .env file into the environment
    load_dotenv()
    path_model = os.getenv('MODEL_PATH')
    path_scaler = os.getenv('SCALER_PATH')

    # Check if the paths were found
    if not path_model or not path_scaler:
        print("Error: MODEL_PATH or SCALER_PATH not found.")
        print("Please ensure your .env file is in the correct directory and is configured properly.")
        exit()

    print(f"Model path found: {path_model}")
    print(f"Scaler path found: {path_scaler}")
    
    # Load model and scaler
    CustomerSegmentor.initialize(path_model, path_scaler)

    # Collect input and predict
    input_tuple = CustomerSegmentor.get_input()
    prediction = CustomerSegmentor.predict(input_tuple)

    print(f"\n The customer belongs to : {prediction}")