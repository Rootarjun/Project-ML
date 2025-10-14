
from fastapi import FastAPI
from pydantic import BaseModel

# Import the 'predict' function from model.py file
from model import predict as predict_segment
from model import __version__ as model_version

# Create a FastAPI application instance
app = FastAPI(
    title="Customer Segmentation API",
    description="An API to predict customer segments using a K-Means model.",
    version= model_version
)

# Define the structure of the input data using Pydantic
class CustomerData(BaseModel):
    age: int
    annual_income: int
    spending_score: int
    gender: int  # 0 for Male, 1 for Female

# Create the API endpoint for prediction
@app.post("/predict")
def handle_prediction(customer_data: CustomerData):
    """
    Receives customer data as a web request and uses the
    imported model to get a prediction.
    """
    # 1. Convert the Pydantic model into a tuple
    input_tuple = (
        customer_data.age,
        customer_data.annual_income,
        customer_data.spending_score,
        customer_data.gender
    )
    
    # 2. Call the imported predict function
    prediction = predict_segment(input_tuple)
    
    # 3. Return the result
    return prediction

# A simple root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"status": "ok", "version": model_version}