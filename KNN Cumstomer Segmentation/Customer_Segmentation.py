import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class CustomerSegmentor:
    """
    A class to assign a customer to a segment using a pre-trained KMeans model and scaler.

    Attributes:
        model (sklearn.cluster.KMeans): The KMeans clustering model loaded from a pickle file.
        scaler (sklearn.preprocessing.StandardScaler): The scaler for standardizing inputs.
    """

    model = None
    scaler = None

    @staticmethod
    def load_pickle(path):
        """
        Load an object from a pickle file.

        Args:
            path (str): File path to the pickle file.

        Returns:
            object: The loaded object.
        """
        with open(path, 'rb') as file:
            return pickle.load(file)

    @classmethod
    def initialize(cls, model_path, scaler_path):
        """
        Initialize the class with a KMeans model and a scaler.

        Args:
            model_path (str): Path to the pickled KMeans model.
            scaler_path (str): Path to the pickled scaler.
        """
        cls.model = cls.load_pickle(model_path)
        cls.scaler = cls.load_pickle(scaler_path)

    @staticmethod
    def get_input():
        """
        Get user input for clustering.

        Returns:
            tuple: (Age, Annual Income, Spending Score, Gender)
        """
        Age = int(input("Enter Age: "))
        Gender = float(input("Enter 0 for Male, 1 for Female: "))
        Annual_income = float(input("Enter Annual income (in k$): "))
        Spending_score = int(input("Enter Spending Score (1–100): "))
        return Age, Annual_income, Spending_score, Gender

    @classmethod
    def predict(cls, input_tuple):
        """
        Predict the customer segment for the input features.

        Args:
            input_tuple (tuple): A tuple (Age, Annual Income, Spending Score, Gender)

        Returns:
            int: Cluster label assigned by KMeans.
        """
        input_array = np.array(input_tuple).reshape(1, -1)
        input_scaled = cls.scaler.transform(input_array)
        prediction = cls.model.predict(input_scaled)
        return prediction[0]


# ===== Main Code =====
if __name__ == "__main__":
    # Paths to the trained model and scaler
    path_model = r'Models/Kmeans.pkl'
    path_scaler = r'Models/KmeansScaler.pkl'

    # Load model and scaler
    CustomerSegmentor.initialize(path_model, path_scaler)

    # Collect input and predict
    input_tuple = CustomerSegmentor.get_input()
    prediction = CustomerSegmentor.predict(input_tuple)

    print(f"The customer belongs to Group: {prediction}")
