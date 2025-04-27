import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class StockPricePredictor:
    """
    A class to predict stock closing prices using a pre-trained model and scaler.

    Attributes:
        model (sklearn model): The machine learning model loaded from a pickle file.
        scaler (sklearn scaler): The scaler loaded from a pickle file for standardization.
    """

    # Static attributes (shared across all instances)
    model = None
    scaler = None

    @staticmethod
    def load_pickle(path):
        """
        Load a pickle file from the given path.

        Args:
            path (str): Path to the pickle file.

        Returns:
            Loaded object from the pickle file.
        """
        with open(path, 'rb') as file:
            return pickle.load(file)

    @classmethod
    def initialize(cls, model_path, scaler_path):
        """
        Initialize the model and scaler by loading them from pickle files.

        Args:
            model_path (str): Path to the saved machine learning model.
            scaler_path (str): Path to the saved scaler.
        """
        cls.model = cls.load_pickle(model_path)
        cls.scaler = cls.load_pickle(scaler_path)

    @staticmethod
    def get_input():
        """
        Collect input features from the user.

        Returns:
            tuple: Opening price, maximum price, minimum price, and volume traded.
        """
        open_price = float(input("Enter Opening Price: "))
        high = float(input("Enter Max Price: "))
        low = float(input("Enter Lowest Price: "))
        vol = int(input("Enter Volume Traded: "))
        return open_price, high, low, vol

    @classmethod
    def predict(cls, input_tuple):
        """
        Predict the closing price for a given input.

        Args:
            input_tuple (tuple): A tuple containing (open, high, low, volume).

        Returns:
            float: Predicted closing price.
        """
        # Ensure input is in the right shape
        input_array = np.array(input_tuple).reshape(1, -1)

        # Standardize the input using the loaded scaler
        input_scaled = cls.scaler.transform(input_array)

        # Predict using the loaded model
        prediction = cls.model.predict(input_scaled)

        return prediction[0]


# ===== Main Code =====
if __name__ == "__main__":
    # Define paths to model and scaler
    path_model = r'Models/LRModel.pkl'
    path_scaler = r'Models/LRScaler.pkl'

    # Initialize model and scaler
    StockPricePredictor.initialize(path_model, path_scaler)

    # Get input from user
    input_tuple = StockPricePredictor.get_input()

    # Predict and display result
    prediction = StockPricePredictor.predict(input_tuple)
    print("The Closing Price for the day is", round(prediction, 3))
