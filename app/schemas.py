from datetime import datetime
from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    tpep_pickup_datetime: datetime
    PULocationID: int
    DOLocationID: int
    trip_distance: float = Field(gt=0)
    passenger_count: float | None = None


class TripResponse(BaseModel):
    predicted_duration_minutes: float