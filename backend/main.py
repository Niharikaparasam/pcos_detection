from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model, load_scaler, predict_pcos
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Load model and scaler once at startup
model = load_model()
scaler = load_scaler()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #if hosted by Flask
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define input data format using Pydantic
class PCOSInput(BaseModel):
    Age: float
    BMI: float
    Cycle: int
    Weight_gain: int
    Hair_growth: int
    Skin_darkening: int
    Hair_loss: int
    Pimples: int
    TSH: float
    Follicle_L: float
    Follicle_R: float


# Home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to PCOS Detection API!"}


# Prediction endpoint
@app.post("/predict/")


def predict(data: PCOSInput):
    input_data = [
        data.Age, data.BMI, data.Cycle, data.Weight_gain,
        data.Hair_growth, data.Skin_darkening, data.Hair_loss,
        data.Pimples, data.TSH, data.Follicle_L, data.Follicle_R
    ]

    prediction = predict_pcos(input_data)

    result = "PCOS Positive" if prediction == 1 else "PCOS Negative"
    print("result: ", result)
    return {"prediction": result}





