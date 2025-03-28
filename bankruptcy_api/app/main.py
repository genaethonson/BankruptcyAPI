# from fastapi import FastAPI, HTTPException
# from app.schema import BankruptcyInput, BankruptcyResponse
# from app.model import predict

# app = FastAPI(title="Bankruptcy Prediction API")

# @app.post("/predict", response_model=BankruptcyResponse)
# def get_prediction(input_data: BankruptcyInput):
#     if len(input_data.features) != 15:
#         raise HTTPException(status_code=400, detail="Exactly 15 features are required.")
    
#     prediction, probability = predict(input_data.features)
#     return BankruptcyResponse(prediction=prediction, probability=probability)
from fastapi import FastAPI
from app.schema import BankruptcyInput, BankruptcyResponse
from app.model import predict

app = FastAPI()

@app.post("/predict", response_model=BankruptcyResponse)
def get_prediction(input_data: BankruptcyInput):
    prediction, probability = predict(input_data.features)
    return BankruptcyResponse(prediction=prediction, probability=probability)

