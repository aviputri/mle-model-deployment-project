import pandas as pd

from features import compute_target, add_time_features, impute_missing, filter_outliers, to_records


def test_compute_target():
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2025-01-01 08:00:00"]),
        "tpep_dropoff_datetime": pd.to_datetime(["2025-01-01 08:15:00"]),
    })
    result = compute_target(df)
    assert result["duration_minutes"].iloc[0] == 15.0


def test_add_time_features():
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2025-01-06 14:30:00"]),  # a Tuesday
    })
    result = add_time_features(df)
    assert result["pickup_hour"].iloc[0] == 14
    assert result["pickup_dayofweek"].iloc[0] == 1  # Monday=0, Tuesday=1


def test_filter_outliers_duration():
    df = pd.DataFrame({
        "duration_minutes": [-5, 12, 500],
        "trip_distance": [2.0, 3.5, 2.0],
    })
    result = filter_outliers(df)
    assert len(result) == 1
    assert result["duration_minutes"].iloc[0] == 12


def test_filter_outliers_distance():
    df = pd.DataFrame({
        "duration_minutes": [10, 10, 10],
        "trip_distance": [-1, 5.0, 999999.0],
    })
    result = filter_outliers(df)
    assert len(result) == 1
    assert result["trip_distance"].iloc[0] == 5.0


def test_impute_missing():
    df = pd.DataFrame({
        "passenger_count": [1.0, None, 2.0, 1.0],
        "PUBorough": ["Manhattan", None, "Queens", "Manhattan"],
        "DOBorough": ["Brooklyn", "Bronx", None, "Brooklyn"],
    })
    result = impute_missing(df)
    assert result["passenger_count"].isna().sum() == 0
    assert result["passenger_count"].iloc[1] == 1.0  # mode (appears twice)
    assert result["PUBorough"].iloc[1] == "Unknown"
    assert result["DOBorough"].iloc[2] == "Unknown"