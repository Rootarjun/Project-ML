import pickle
import numpy as np
import os
import warnings
from dotenv import load_dotenv 
warnings.filterwarnings("ignore")

__version__="0.1.0"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 2. Load environment variables
load_dotenv()
path_model_relative = os.getenv('MODEL_PATH') 
path_scaler_relative=os.getenv('SCALER_PATH')

path_model=os.path.join(BASE_DIR, path_model_relative)
path_scaler=os.path.join(BASE_DIR, path_scaler_relative)

with open(path_model, 'rb') as file:
    model=pickle.load(file)

with open(path_scaler, 'rb') as file:
    scaler=pickle.load(file)
    
def predict(input_tuple):
    """Predict the customer segment for the input features."""
    
    input_array = np.array(input_tuple).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

        
        
    descriptive_labels = [
            "Group 1 - Target Customers (High Income, High Spending)",
            "Group 2 - Careful Spenders (High Income, Low Spending)",
            "Group 3 - Standard Customers (Average Income, Average Spending)",
            "Group 4 - Frugal Customers (Low Income, Low Spending)",
            "Group 5 - Careless Spenders - Young People(Low Income, High Spending)",
            "Group 6  - Young People - Working Profesionals (Moderate Income, Moderate Spending)"
        ]

        
    return descriptive_labels[prediction[0]]
#age, annual income, spending score, gender
print(predict([20,10,1,1]))