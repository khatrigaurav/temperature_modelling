"""
Created on : Nov 18, 2023

@author: Gaurav
@version: 1.0

OOP implementation of the predictor to see how easy it is to use

"""

import os
import time
import sys
import pandas as pd
import numpy as np
import pickle
import geopandas as gpd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


from shapely.geometry import Point
from pyproj import CRS

from lib import helper as helper
from lib import visualizer

MODEL_PATH = '/Users/gaurav/UAH/temperature_modelling/Resources/trained_models/'
PREDICTION_DIR = '/Users/gaurav/UAH/temperature_modelling/Resources/predictions/Madison_2021'

TEST_ECOSTRSS_FILE = '/Users/gaurav/UAH/temperature_modelling/data/raster_op/ECOSTRESS_values_testing.csv'
TEST_URBAN_FILE = '/Users/gaurav/UAH/temperature_modelling/data/raster_op/urban_surface_properties_values_testing.csv'
time_adjusted_df = pd.read_csv('/Users/gaurav/UAH/temperature_modelling/Analytics/time_adjusted_df.csv')
closest_station_csv_path = '/Users/gaurav/UAH/temperature_modelling/Analytics/'

# this file is generated in training to get list of columns used for training
#This renamed to model_path+modeL_name+'_cols_list.csv'
gen_test_data_file = pd.read_csv('Analytics/X_test.csv').head(1)


def initialize_closest_station(ec_data):
    ''' This function initializes the closest station csv file
        Creating this file again and again is too slow, so we will create it once and then use it
    '''
    if not os.path.exists(os.path.join(closest_station_csv_path, 'closest_stations.csv')):
        print('Closest station file not found, Creating closest station csv file')
        closest_station_data = helper.find_closest_station(time_adjusted_df[time_adjusted_df.hour == 12],ec_data)
        closest_station_data.to_csv(os.path.join(closest_station_csv_path, 'closest_stations.csv'), index=False)

    else:
        print('Closest station file found, Reading closest station csv file')
        closest_station_data = pd.read_csv(os.path.join(closest_station_csv_path, 'closest_stations.csv'))
    
    return closest_station_data

def get_station_ranges(len_ec_data):
    ''' This function returns the station ranges for sampling of data

        args: len_ec_data: length of the ec_data : len(ec_data.station.unique())
    '''

    stations_ranges = []
    max_range = len_ec_data//30000
    start_range = 1
    end_range = 30001
    for i in range(max_range):
        station_range = np.arange(start_range, end_range)
        start_range = end_range
        end_range = end_range+30000
        stations_ranges.append(station_range)

    # this list to be used for sampling of data
    stations_ranges.append(np.arange(start_range, len_ec_data+1))

    return stations_ranges


def get_ec_urban(ec_data=None, urb_data=None):
    ''' This function returns the ECOSTRESS and Urban Surface data for the given hour
        Parameters : [1] or None

        Rule : This function will never be called from class, because reading the entire ecostress file each time is slow

    '''

    if ec_data is None:
        ec_data = pd.read_csv(TEST_ECOSTRSS_FILE)

    if urb_data is None:
        urb_data = pd.read_csv(TEST_URBAN_FILE)

    # ec_data = ec_data.query(
    #     f'hour == {list([hour_filter or list(np.arange(0,24,1))][0])}')

    ec_data = gpd.GeoDataFrame(ec_data)
    ec_data['geometry'] = [Point(xy) for xy in zip(
        ec_data['longitude'], ec_data['latitude'])]
    ec_data.crs = CRS.from_epsg(6879).to_wkt()
    ec_data = ec_data.to_crs(epsg=4326)
    ec_data['latitude'] = ec_data.geometry.y
    ec_data['longitude'] = ec_data.geometry.x

    urb_data = gpd.GeoDataFrame(urb_data)
    urb_data['geometry'] = [Point(xy) for xy in zip(
        urb_data['longitude'], urb_data['latitude'])]
    urb_data.crs = CRS.from_epsg(6879).to_wkt()
    urb_data = urb_data.to_crs(epsg=4326)
    urb_data['latitude'] = urb_data.geometry.y
    urb_data['longitude'] = urb_data.geometry.x

    # #Making sure of the precision for proper joins
    ec_data['latitude'] = np.round(ec_data['latitude'], 7)
    ec_data['longitude'] = np.round(ec_data['longitude'], 7)

    urb_data['latitude'] = np.round(urb_data['latitude'], 7)
    urb_data['longitude'] = np.round(urb_data['longitude'], 7)

    ec_data = ec_data[['station', 'latitude',
                       'longitude', 'hour', 'value_LST']]
    urb_data = urb_data[['station', 'latitude', 'longitude', 'valueImperviousfraction',
                         'valueTreefraction', 'valueBuildingheight', 'valueNearestDistWater', 'valueWaterfraction', 'valueBuildingfraction']]

    station_ranges = get_station_ranges(len(ec_data.station.unique()))

    return ec_data, urb_data, station_ranges

def merge_predictions(predict_dir):
    ''' It merges all the predictions into one file '''
    files = os.listdir(predict_dir)
    files = sorted([os.path.join(predict_dir, x)
                   for x in files if 'final' in x])

    df1 = pd.read_csv(files[0])

    for file in files[1:]:
        df2 = pd.read_csv(file)
        df1 = pd.concat([df1, df2])

    df1 = helper.convert_to_gpd(df1, 'epsg:4326', convert_to='epsg:6879')
    df1 = df1.groupby(['latitude', 'longitude']).mean().reset_index()

    # change value less than 0 to 0
    df1['prediction_temp'] = df1['prediction_temp'].apply(
        lambda x: 0 if x < -30 else x)

    return df1



@dataclass
class Model:
    ''' 
    Model class attributes to be used for prediction
    '''
    model_name: str                             # Name of the model to be used  
    model_path: str                            # Path where the model is stored     
    model_hour_filter: int                   # Hour filter to be used for prediction

    model_class: str = field(init=False)              # Model class to be used for prediction (eg. LinearRegression, RandomForestRegressor, etc.)
    model_predict_dir: str = field(init=False)       # Directory where the predictions will be stored
    model_closest_flag: bool = field(init=False)    # Flag to check if closest station temperature is used in the model
    model_column_list: list = field(init=False)           # List of columns to be used for prediction


    def __post_init__(self):
        self.model_class = self.model_name.split('_')[0]      
        self.model_predict_dir = os.path.join(PREDICTION_DIR, self.model_class)   
        # self.model_closest_flag = 'closest_station_1_temp' in gen_test_data_file.columns  
        self.model_column_list  = pd.read_csv(os.path.join(self.model_path,self.model_class,self.model_class+'_cols_list.csv')).columns.to_list()
        self.model_closest_flag = 'closest_station_1_temp' in self.model_column_list



class Predictor():
    ''' 
    Predictor class to predict temperature values based on ECOSTRESS and urban surface data
    '''

    def __init__(self, model_name, hour_filter=None):
        if hour_filter is None:
            print('No hour filter provided, provide valid hour ranges')
            sys.exit()
        

        self.model_attrs = Model(model_name, MODEL_PATH, hour_filter)
        self.model = pickle.load(
            open(os.path.join(self.model_attrs.model_path ,self.model_attrs.model_class ,self.model_attrs.model_name), 'rb'))  # Loading the model
        
        #creating the preditcion directory if it does not exist
        os.makedirs(self.model_attrs.model_predict_dir, exist_ok=True)

        # Time adjusted dataframe to be used for prediction
        self.time_adjusted_df = time_adjusted_df
        self.model_hour_flag = 'hour' in gen_test_data_file.columns
        self.ec_data_seg = None
        self.closest_station_data = None
        self.column_list = pd.read_csv(os.path.join(self.model_attrs.model_path,self.model_attrs.model_class,self.model_attrs.model_class+'_cols_list.csv')).columns.to_list()


    def set_ec_segments(self, ec_data):
        '''
        Segments the ecostress and urban surface data for the given hour
        '''
        self.ec_data_seg = ec_data.query('hour == @self.model_attrs.model_hour_filter')

        return self.ec_data_seg
    
    
    def add_temp_data(self,closest_station_data):
        '''
        This function adds the temperature data to the dataframe
        Returns : Station and temperature data for the closest stations

        '''
        time_adjusted_df_eco_merge = time_adjusted_df[[
            'station', 'latitude', 'longitude','temperature','hour']].drop_duplicates()

        time_adjusted_df_eco_merge = time_adjusted_df_eco_merge.query('hour == @self.model_attrs.model_hour_filter')
        closest_stations_1 = closest_station_data[[
            'station', 'closest_station_1', 'closest_1_distance', 'latitude', 'longitude','hour']]
        closest_stations_2 = closest_station_data[[
            'station', 'closest_station_2', 'closest_2_distance', 'latitude', 'longitude','hour']]
        # closest_stations_3 = closest_stations[['station', 'closest_station_3', 'closest_3_distance','latitude','longitude']]

        x1 = pd.merge(closest_stations_1, time_adjusted_df_eco_merge[['station', 'temperature']],
                    left_on='closest_station_1', right_on='station', how='left')
        x1 = x1.rename({'station_x': 'station', 'temperature': 'closest_station_1_temp'}, axis=1).drop(
            ['station_y'], axis=1)
        x1 = x1.fillna(method='ffill')

        x2 = pd.merge(closest_stations_2, time_adjusted_df_eco_merge[['station', 'temperature']],
                    left_on='closest_station_2', right_on='station', how='left')
        x2 = x2.rename({'station_x': 'station', 'temperature': 'closest_station_2_temp'}, axis=1).drop(
            ['station_y'], axis=1)
        x2 = x2.fillna(method='ffill')

        x2 = x2[['station', 'hour', 'latitude', 'longitude',
                'closest_station_2_temp', 'closest_2_distance']]

        x1 = x1[['station', 'hour', 'latitude', 'longitude', 'closest_station_1_temp',
                'closest_1_distance']].drop_duplicates()
        x2 = x2[['station', 'hour', 'latitude', 'longitude', 'closest_station_2_temp',
                'closest_2_distance']].drop_duplicates()

        final = x1.merge(x2, on=['station', 'hour', 'latitude',
                        'longitude'], how='inner')
        
        return final

    def add_urban_data(self,ec_data_segment, urb_data_segment, closest_columns=None):
        ''' 
        This function adds the urban data to the final dataframe thats calculated in calculate_predictions()
        Feed only segment of ec_data
        '''
        urb_cols = ['valueImperviousfraction', 'valueTreefraction', 'valueBuildingheight',
                    'valueNearestDistWater', 'valueWaterfraction', 'valueBuildingfraction']
        lst_cols = ['value_LST']

        urb_merged = ec_data_segment.merge(
            urb_data_segment, on=['station', 'latitude', 'longitude'], how='inner')
        

        urb_merged = urb_merged[['station', 'latitude',
                                'longitude', 'hour'] + lst_cols + urb_cols]

        urb_merged['latitude']  = np.round(urb_merged['latitude'], 7)
        urb_merged['longitude'] = np.round(urb_merged['longitude'], 7)


        if closest_columns is not None:
            closest_columns['latitude']  = np.round(closest_columns['latitude'], 7)
            closest_columns['longitude']  = np.round(closest_columns['longitude'], 7)


            test_data = closest_columns.merge(
                urb_merged, on=['station', 'latitude', 'longitude', 'hour'], how='inner')
            test_data = test_data.drop('station', axis=1).rename(
                {'value_LST': 'adjusted_lst'}, axis=1)
            
            # test_data['day_of_year'] = test_data['beg_time'].dt.dayofyear
            # test_data['day_of_year'] = pd.to_datetime(test_data['beg_time']).dt.dayofyear

        else:
            test_data = urb_merged.rename({'value_LST': 'adjusted_lst'}, axis=1)
            # test_data['day_of_year'] = 1


        if 'value_LST_x' in test_data.columns:
            test_data = test_data.drop('value_LST_y', axis=1).rename(
                {'value_LST_x': 'adjusted_lst'}, axis=1)

        if 'closest_1_distance_y' in test_data.columns:
            test_data = test_data.drop('closest_1_distance_y', axis=1)

        # handling null values
        test_data['valueBuildingheight'] = test_data['valueBuildingheight'].fillna(
            0)
        test_data['adjusted_lst'] = test_data['adjusted_lst'].fillna(
            test_data['adjusted_lst'].mean())
        test_data['valueNearestDistWater'] = 1 / \
            (1 + test_data['valueNearestDistWater'])
        test_data.loc[test_data.valueTreefraction < -10, 'valueTreefraction'] = 0

        return test_data

    def calculate_predictions(self, urb_data, stations_ranges,debug=False):
        ''' Function to calculate predictions for all stations in the dataset
            Returns a list of columns that's necessary for creating plots
        '''
        # col_file_name = os.path.join(MODEL_PATH, MODEL_CLASS+'_cols_list.csv')
        # COL_LIST = pd.read_csv(col_file_name).columns.tolist()

        # col_list = gen_test_data_file.columns.tolist()
        col_list = self.model_attrs.model_column_list
        # print(col_list)
        timex = time.time()
        if self.model_attrs.model_closest_flag:
            "This could be read only once"
            closest_station_data = initialize_closest_station(self.ec_data_seg)

        for index, station_list in enumerate(stations_ranges[:len(stations_ranges)]):
            time1 = time.time()
            ec2_segment = self.ec_data_seg[self.ec_data_seg.station.isin(station_list)]
            urb2_segment = urb_data[urb_data.station.isin(station_list)]


            if self.model_attrs.model_closest_flag:
                # closest_station_data = initialize_closest_station(self.ec_data_seg)

                #The filter is set to @closest_station_data.hour.unique()[0] because closest stations are independent of hour
                closest_station_data = closest_station_data.query('hour == @closest_station_data.hour.unique()[0]')

                #It contains the closest station data for the given hour
                final = self.add_temp_data(closest_station_data)
                test_data = self.add_urban_data(ec2_segment, urb2_segment, closest_columns=final)
                if debug:
                    return ec2_segment, urb2_segment,test_data

            else:
                test_data = self.add_urban_data(ec2_segment, urb2_segment, closest_columns=None)

            test_data = test_data.fillna(method='ffill')
            # print(test_data.columns)
            # print(col_list)
            # print(f'Column difference : {set(col_list) - set(test_data.columns)}')
            if debug:
                print(test_data.head())
                print(test_data.columns)
                return
            
            # print(f'Column difference : {set(col_list) - set(test_data.columns)}')
            # print(test_data.columns)
            test_data['prediction_temp'] = self.model.predict(test_data[col_list])
            column_set = ['latitude', 'longitude']
            
            if 'hour' in test_data.columns:
                column_set.append('hour')
            save_df = test_data[column_set + ['prediction_temp', 'adjusted_lst']]
            save_df = save_df.groupby(column_set).mean().reset_index()
            save_df['station'] = station_list

            save_df.to_csv(f'{self.model_attrs.model_predict_dir}/final_preds_{index}.csv', index=False)
            # print(f'Time taken to save model  data : {round(time.time()-time1,2)} seconds')
            # print(
            #     f'Iteration {index+1}/{len(stations_ranges)} complete in {round(time2-time1,2)} seconds')
            # print("#############################################")

        print(f'Predictions complete, saved in {self.model_attrs.model_predict_dir}')
        print(f'Time Taken : {round(time.time()-timex,2)} seconds)')
        test_col_list = test_data[col_list].columns.to_list()
        
        return test_col_list

    def get_rasters(self, urb_data, stations_ranges,debug=False):
        ''' Create rasters for the predictions and the adjusted LST
        Returns : Raster[1,2], bounds and test_column_list
                First raster is of predictions and second is of adjusted LST
        '''

        test_column_list = self.calculate_predictions(urb_data, stations_ranges)
        df = merge_predictions(self.model_attrs.model_predict_dir)

        raster1, bounds = visualizer.get_raster(df, 'prediction_temp')
        raster2, bounds = visualizer.get_raster(df, 'adjusted_lst')

        return [raster1, raster2], bounds, test_column_list


    def runner(self, urb_data, stations_ranges, debug=False):
        '''Wrapper for calculate_predictions and get_rasters
        '''

        return self.get_rasters(urb_data, stations_ranges, debug)

