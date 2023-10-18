"""
Created on : Oct 18, 2023

@author: Gaurav
@version: 1.0

Helper functions that are independent of the source and help in handling dataframes.

Rule : They must take dataframe as one input and return a dataframe as output. Exception: calculate_distance

Functions:
        Resample_daily
        run_mean_comparison --> main
            find_closest
                calculate_distance
            find_anomaly


"""
from geopy.distance import geodesic
import pandas as pd
import numpy as np


def calculate_distance(coord1, coord2):
    """Calculates the distance between two coordinates in km"""
    return geodesic(coord1, coord2).kilometers


def resample_daily(df):
    """Resamples the dataframe to daily values : Previous animation plot was hourly, need to do it daily now"""

    df = df.sort_values(by=["station", "beg_time"])
    df.index = [df.beg_time, df.station]
    df = df.groupby(
        [pd.Grouper(level="station"), pd.Grouper(level="beg_time", freq="D")]
    ).mean()

    df = df.reset_index()
    df["beg_time"] = pd.to_datetime(df.beg_time)
    df.sort_values(by=["station", "beg_time", "day_of_year"], inplace=True)

    df = df[
        ["station", "beg_time", "temperature",
            "day_of_year", "latitude", "longitude"]
    ]
    df["temperature"] = df.temperature.round(2)

    return df


def find_closest(df):
    """
    This function finds 3 closest stations for each station in the dataframe
    """

    df["latitude"] = round(df["latitude"], 5)
    df["longitude"] = round(df["longitude"], 5)

    stations = (
        df.groupby("station")["latitude",
                              "longitude"].value_counts().reset_index()
    ).drop(0, axis=1)

    # Initialize an empty DataFrame to store the closest stations
    closest_stations_df = pd.DataFrame(
        columns=[
            "station",
            "closest_station_1",
            "closest_station_2",
            "closest_station_3",
        ]
    )

    # Create empty lists to store results
    station_list = []
    closest_station_1_list = []
    closest_station_2_list = []
    closest_station_3_list = []

    for each_station in stations["station"]:
        # Create a list to store distances
        distance_list = []

        # Create a list to store the closest stations
        closest_stations = []

        # Get the coordinates of the station
        station_coord = (
            stations[stations["station"] == each_station][
                ["latitude", "longitude"]
            ].values[0][0],
            stations[stations["station"] == each_station][
                ["latitude", "longitude"]
            ].values[0][1],
        )

        # Calculate the distance between the station and all other stations
        for each_row in stations.iterrows():
            distance_list.append(
                calculate_distance(
                    station_coord, (each_row[1]["latitude"],
                                    each_row[1]["longitude"])
                )
            )

        # Sort the distances and get the three closest stations
        closest_stations = np.array(distance_list).argsort()[:4]

        # Append the results to the lists
        station_list.append(each_station)
        closest_station_1_list.append(
            stations.iloc[closest_stations[1]]["station"])
        closest_station_2_list.append(
            stations.iloc[closest_stations[2]]["station"])
        closest_station_3_list.append(
            stations.iloc[closest_stations[3]]["station"])

    # Create the DataFrame using the lists
    closest_stations_df = pd.DataFrame(
        {
            "station": station_list,
            "closest_station_1": closest_station_1_list,
            "closest_station_2": closest_station_2_list,
            "closest_station_3": closest_station_3_list,
        }
    )

    return closest_stations_df


# df[(df.station == "KWIWINDS10" )and (df.beg_time > "2020-05-01")]


def find_anomaly(closest_, df_daily, anomaly_threshold=0.6):
    """Returns the dataframe that consist of anomalous days

    TODO : Maybe we can replace it with something like dynamic time warping (DTW algorithm) instead of static mean comparison
    """

    df_slice = df_daily.join(closest_.set_index("station"), on="station")

    df_slice = df_slice[
        [
            "station",
            "beg_time",
            "temperature",
            "closest_station_1",
            "closest_station_2",
            "closest_station_3",
        ]
    ]

    df_joined = pd.merge(
        df_slice[
            [
                "station",
                "temperature",
                "beg_time",
                "closest_station_1",
                "closest_station_2",
                "closest_station_3",
            ]
        ],
        df_daily[["station", "beg_time",
                        "latitude", "longitude", "temperature"]],
        left_on=["closest_station_1", "beg_time"],
        right_on=["station", "beg_time"],
        how="left",
    )
    df_joined = df_joined.rename(
        columns={
            "station_x": "station",
            "temperature_x": "temperature",
            "temperature_y": "closest_station_1_temp",
        }
    ).drop("station_y", axis=1)

    df_joined = pd.merge(
        df_joined[
            [
                "station",
                "temperature",
                "beg_time",
                "closest_station_1_temp",
                "closest_station_1",
                "closest_station_2",
                "closest_station_3",
            ]
        ],
        df_daily[["station", "beg_time",
                        "latitude", "longitude", "temperature"]],
        left_on=["closest_station_2", "beg_time"],
        right_on=["station", "beg_time"],
        how="inner",
    )
    df_joined = df_joined.rename(
        columns={
            "station_x": "station",
            "temperature_x": "temperature",
            "temperature_y": "closest_station_2_temp",
        }
    ).drop("station_y", axis=1)

    df_joined = pd.merge(
        df_joined[
            [
                "station",
                "temperature",
                "beg_time",
                "closest_station_1_temp",
                "closest_station_2_temp",
                "closest_station_1",
                "closest_station_2",
                "closest_station_3",
            ]
        ],
        df_daily[["station", "beg_time",
                        "latitude", "longitude", "temperature"]],
        left_on=["closest_station_3", "beg_time"],
        right_on=["station", "beg_time"],
        how="left",
    )
    df_joined = df_joined.rename(
        columns={
            "station_x": "station",
            "temperature_x": "temperature",
            "temperature_y": "closest_station_3_temp",
        }
    ).drop("station_y", axis=1)

    df_joined = df_joined.sort_values(by=["station", "beg_time"])
    df_joined["average_temperature"] = np.mean(
        df_joined[
            [
                "closest_station_1_temp",
                "closest_station_2_temp",
                "closest_station_3_temp",
            ]
        ],
        axis=1,
    )

    df_joined = df_joined[
        [
            "station",
            "beg_time",
            "temperature",
            "average_temperature",
            "latitude",
            "longitude",
        ]
    ]

    df_joined["difference"] = (
        abs(df_joined["temperature"] - df_joined["average_temperature"])
        / df_joined["temperature"]
    )
    anamoly = df_joined[df_joined["difference"].apply(lambda x: x > anomaly_threshold)]
    anamoly = anamoly[["station", "beg_time","temperature", "average_temperature"]]

    return anamoly


def run_mean_comparison(df,anomaly_threshold=0.5):
    """Runs the mean comparison for the dataframe and returns the anomalous days"""

    df_daily = resample_daily(df)
    closest_ = find_closest(df_daily)

    anamoly = find_anomaly(closest_, df_daily,anomaly_threshold=anomaly_threshold)

    return anamoly


# Usage :

# import numpy as np


# df_daily = resample_daily(df_vegas)
# closest_ = find_closest(df_daily)

# anamoly = find_anomaly(closest_,df_daily)
