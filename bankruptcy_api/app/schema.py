from pydantic import BaseModel
from typing import List

class BankruptcyInput(BaseModel):
    features: List[float]

class BankruptcyResponse(BaseModel):
    prediction: int
    probability: float
