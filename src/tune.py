"""
Hyperparameter tuning for the RandomForestRegressor.
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from train import MLFLOW_DB_PATH, RANDOM_STATE, build_features, load_data, save_model, save_vectorizer

PARAM_DISTRIBUTIONS = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [10, 15, 20, 25, 30, None],
    "min_samples_leaf": [1, 2, 5, 10, 20],
}

SEARCH_SAMPLE_SIZE = 300_000


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("nyc-taxi-duration")

    print("Loading data...")
    df, zones = load_data()

    print("Building features...")
    X_train, X_val, y_train, y_val, dv = build_features(df, zones)

    # search on a subsample for speed
    sample_idx = range(min(SEARCH_SAMPLE_SIZE, X_train.shape[0]))
    X_search = X_train[sample_idx]
    y_search = y_train[sample_idx]

    print(f"Running RandomizedSearchCV on {X_search.shape[0]} rows...")
    base_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=10,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
    )
    search.fit(X_search, y_search)
    print(f"Best params found: {search.best_params_}")

    print("Retraining best params on the FULL training set...")
    best_model = RandomForestRegressor(**search.best_params_, random_state=RANDOM_STATE, n_jobs=-1)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    print(f"Tuned model validation RMSE: {rmse:.3f}")

    with mlflow.start_run(run_name="tuned_rf"):
        mlflow.log_params(search.best_params_)
        mlflow.log_param("search_sample_size", X_search.shape[0])
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(
            best_model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.log_artifact(str(save_vectorizer(dv)))
        mlflow.log_artifact(str(save_model(best_model)))


if __name__ == "__main__":
    main()