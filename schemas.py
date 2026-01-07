from pydantic import BaseModel

class InsuranceInput(BaseModel):
    Age: int
    Vintage: int
    Annual_Premium: float
    Gender: str
    Vehicle_Age: str
    Vehicle_Damage: str
    Driving_License: int
    Region_Code: int
    Previously_Insured: int
    Policy_Sales_Channel: int


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
