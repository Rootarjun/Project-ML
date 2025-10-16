# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware # 1. Import CORS Middleware
from model import predict as predict_segment

__version__ = "0.1.0"

app = FastAPI(
    title="Customer Segmentation API",
    description="An API to predict customer segments using a K-Means model.",
    version=__version__
)

# 2. Add the CORS middleware to your app
origins = ["*"] # Allow all origins for simplicity in local development

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

class CustomerData(BaseModel):
    age: int
    annual_income: int
    spending_score: int
    gender: int

@app.post("/predict")
def handle_prediction(customer_data: CustomerData):
    input_tuple = (
        customer_data.age,
        customer_data.annual_income,
        customer_data.spending_score,
        customer_data.gender
    )
    prediction = predict_segment(input_tuple)
    return {"predicted_segment": prediction}

@app.get("/")
def read_root():
    return FileResponse('index.html')