"""
Data preprocessing for the NYC taxi duration prediction project.
"""
#%%
# import

import pandas as pd

#%% 
# lookup table for the NYC taxi zone ID boroughs
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# feature in strings and in integer
CATEGORICAL = ["PUBorough", "DOBorough"]
NUMERICAL = ["trip_distance", "passenger_count", "pickup_hour", "pickup_dayofweek"]

#%%
# functions

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    #Add duration_minutes, computed from tpep_pickup/dropoff_datetime.
    df["duration_minutes"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    #Add pickup_hour and pickup_dayofweek, derived from tpep_pickup_datetime.
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    
    return df


def add_borough_features(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    #Merge PULocationID/DOLocationID against the zone lookup to get PUBorough/DOBorough.
    df = df.merge(
        zones[["LocationID", "Borough"]].rename(columns={"Borough": "PUBorough"}), 
        left_on="PULocationID", right_on="LocationID", how="left"
        ).drop(columns=["LocationID"])

    df = df.merge(
        zones[["LocationID", "Borough"]].rename(columns={"Borough": "DOBorough"}),
        left_on="DOLocationID", right_on="LocationID", how="left"
        ).drop(columns=["LocationID"])
    
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    #Apply NA handling (passenger_count, PUBorough, DOBorough)
    
    #passenger_count impute with the most common value
    df["passenger_count"] = df["passenger_count"].fillna(df["passenger_count"].mode()[0])

    #PUBorough and DOBorough impute with "Unknown"
    df["PUBorough"] = df["PUBorough"].fillna("Unknown")
    df["DOBorough"] = df["DOBorough"].fillna("Unknown")

    return df

def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    #outlier filters (duration_minutes, trip_distance).
    #duration_minutes filter: remove trips with duration < 1 minute or > 60 minutes
    df = df[(df["duration_minutes"] >= 1) & (df["duration_minutes"] <= 60)]
    #trip_distance filter: remove trips with distance < 0 miles or > 100 miles
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] <= 100)]
    
    return df

def prepare_dataframe(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    #Run the full pipeline: target, time features, borough features, cleaning.
    df = compute_target(df)
    df = add_time_features(df)
    df = add_borough_features(df, zones)
    df = impute_missing(df)
    df = filter_outliers(df)
    
    return df

def prepare_for_prediction(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_borough_features(df, zones)
    df = impute_missing(df)
    
    return df

def to_records(df: pd.DataFrame) -> list[dict]:
    #Select CATEGORICAL + NUMERICAL columns and convert to a list of dicts for DictVectorizer.
    df[CATEGORICAL] = df[CATEGORICAL].astype(str)
    
    return df[CATEGORICAL + NUMERICAL].to_dict(orient="records")
