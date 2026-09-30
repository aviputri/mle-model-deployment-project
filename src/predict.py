import pandas as pd

from features import prepare_for_prediction, to_records


def predict_duration(request_data: dict, model, dv, zones) -> float:
    df = pd.DataFrame([request_data])
    df = prepare_for_prediction(df, zones)
    records = to_records(df)

    X = dv.transform(records)
    prediction = model.predict(X)[0]
    return float(prediction)