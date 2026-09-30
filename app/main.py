import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import skops.io as sio
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # lets us import features.py

from features import ZONE_LOOKUP_URL, prepare_for_prediction, to_records
from app.schemas import TripRequest, TripResponse

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

model = None
dv = None
zones = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, dv, zones
    model_path = ARTIFACTS_DIR / "model.skops"
    dv_path = ARTIFACTS_DIR / "dict_vectorizer.skops"

    model = sio.load(model_path, trusted=sio.get_untrusted_types(file=model_path))
    dv = sio.load(dv_path, trusted=sio.get_untrusted_types(file=dv_path))
    zones = pd.read_csv(ZONE_LOOKUP_URL)
    yield


app = FastAPI(title="NYC Taxi Duration Predictor", lifespan=lifespan)


@app.post("/predict", response_model=TripResponse)
def predict(request: TripRequest):
    print(f"Received request: {request.model_dump()}")
    df = pd.DataFrame([request.model_dump()])
    df = prepare_for_prediction(df, zones)
    records = to_records(df)

    X = dv.transform(records)
    prediction = model.predict(X)[0]

    print(f"Predicted duration: {prediction:.2f} minutes")
    return TripResponse(predicted_duration_minutes=float(prediction))