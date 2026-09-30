"""Train and log a RandomForestRegressor for NYC Yellow Taxi trip duration."""
#run with: uv run python src/train.py

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import skops.io as sio
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve

from features import ZONE_LOOKUP_URL, prepare_dataframe, to_records

# --- paths, anchored to the project root so this works regardless of cwd ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "yellow_tripdata_2025-01.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

RANDOM_STATE = 42
TEST_SIZE=0.3

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
)

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    #Load the raw trip data and the zone lookup table
    if DATA_PATH.exists():
        print(f"Reusing existing file: {DATA_PATH}")
    else:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL}")
        urlretrieve(DATA_URL, DATA_PATH)
        print(f"Saved file to {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    zones = pd.read_csv(ZONE_LOOKUP_URL)
    
    return df, zones


def build_features(df: pd.DataFrame, zones: pd.DataFrame):
    #Run prepare_dataframe, split, and vectorize. Returns X_train, X_val, y_train, y_val, dv
    
    df = prepare_dataframe(df, zones)
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_dicts, val_dicts = to_records(train_df), to_records(val_df)

    # fit ONLY on train, then transform both
    dv = DictVectorizer()
    #features
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    #target
    y_train = train_df["duration_minutes"].values
    y_val = val_df["duration_minutes"].values

    return X_train, X_val, y_train, y_val, dv


def train_model(X_train, y_train, params: dict) -> RandomForestRegressor:
    #Fit a RandomForestRegressor with the given params.
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def evaluate(model: RandomForestRegressor, X_val, y_val) -> float:
    #Return validation RMSE
    y_pred = model.predict(X_val)
    
    return root_mean_squared_error(y_val, y_pred)



def save_vectorizer(dv: DictVectorizer) -> Path:
    #Persist the fitted DictVectorizer with skops, return the saved path.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "dict_vectorizer.skops"
    sio.dump(dv, path)
    return path

def save_model(model: RandomForestRegressor) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "model.skops"
    sio.dump(model, path)
    return path

def main():
    print("Setting up MLflow tracking...")
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("nyc-taxi-duration")

    print("Loading data...")
    df, zones = load_data()

    print("Building features (cleaning, splitting, vectorizing)...")
    X_train, X_val, y_train, y_val, dv = build_features(df, zones)

    params = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run():
        print("Training model...")
        model = train_model(X_train, y_train, params)
        
        print("Evaluating model...")
        rmse = evaluate(model, X_val, y_val)
        print(f"Validation RMSE: {rmse:.3f}")

        print("Logging params, metrics, and model to MLflow...")
        mlflow.log_params(params)
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print("Saving vectorizer and model to artifacts/...")
        vectorizer_path = save_vectorizer(dv)
        mlflow.log_artifact(str(vectorizer_path))
        model_path = save_model(model)
        mlflow.log_artifact(str(model_path))

        print("Done.")


if __name__ == "__main__":
    main()