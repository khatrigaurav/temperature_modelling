"""
Created on : Dec 5, 2023

@author: Gaurav
@version: 1.0

The sole purpose of this file is to train multiple models, predict the results and save the results at once
to do quick analysis.

"""


#Custom modules

import os
import pickle
import datetime
import time
import numpy as np
import pandas as pd
import rioxarray as rxr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import polars as pl #libs for faster data processing


from dateutil import tz
import geopandas as gpd
from shapely.geometry import Point

import sklearn.metrics as metrics
from sklearn.model_selection import  train_test_split

from sklearn.linear_model import LinearRegression


#Custom modules
import satellite as sat
import dataprocess as dp 
import helper
import modeller as mod
import predictor
import visualizer as visualizer

from pyproj import CRS

from sklearn.preprocessing import StandardScaler

window_size = 5
#Its here because I don't want to pass final_df everytime
def get_partitions(final_df,col_list,selection_hour=None,scaler=False):
    ''' Get final split of data based on hour selected
        selection_hour = [1] or None
    '''

    final_df_x = final_df.query(f'hour == {list([selection_hour or list(np.arange(0,24,1))][0])}')
    X_train, X_test, y_train, y_test = mod.get_train_test_data(final_df_x,window_size)

    d_train, d_test = X_train[col_list], X_test[col_list]

    #if there is no hour column, then its set to none, such that the plotter function behaves accordingly
    hour_status = True if 'hour' in d_train.columns else False

    if scaler:
        scaler = StandardScaler()

        d_train =  scaler.fit_transform(d_train)
        d_train = pd.DataFrame(d_train,columns=col_list)
        d_test = scaler.transform(d_test)
        d_test = pd.DataFrame(d_test,columns=col_list)

    return [d_train, d_test, y_train, y_test], hour_status

col_list = [
            # 'latitude', 'longitude', 
            'hour', 
            # 'closest_station_1_temp',
            # 'closest_1_distance',
            #  'day_of_year', 
            'adjusted_lst',
            'valueImperviousfraction', 'valueTreefraction', 'valueBuildingheight',
             'valueNearestDistWater',
            'valueWaterfraction', 'valueBuildingfraction']

grouped_data = pd.read_csv('Analytics/temp_data/grouped_data_untouched.csv')

# should come from argument
modelx = LinearRegression()
MODEL_PATH    = '/Users/gaurav/UAH/temperature_modelling/Resources/trained_models/'
model_name = modelx.__class__.__name__
model_path = os.path.join(MODEL_PATH,model_name)

import glob

files = glob.glob(model_path)
for f in files:
    os.remove(f)


# error_dict = {}
# for i in range(0,24,1):
#     print(f'Hour : {i}')
#     data,hour_status = get_partitions(grouped_data,col_list,[i])  #or None
#     error = mod.train_save(modelx, data, hour_flag=hour_status,bulk_mode=True)
#     error_dict[i] = error

