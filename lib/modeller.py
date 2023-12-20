
"""
Created on : Nov 18, 2023

@author: Gaurav
@version: 1.0

Functions to be used in 00 Modeller.ipynb
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pickle
import time

from sklearn.model_selection import train_test_split
from lib import satellite as sat
from lib import helper as helper
from lib import predictor
from lib import visualizer




MODEL_PATH    = '/Users/gaurav/UAH/temperature_modelling/Resources/trained_models/'
TRAIN_ECOSTRSS_FILE = '/Users/gaurav/UAH/temperature_modelling/data/raster_op/ECOSTRESS_values.csv'
TRAIN_URBAN_FILE = '/Users/gaurav/UAH/temperature_modelling/data/raster_op/urban_surface_properties_values.csv'


def create_satellite_data(location='Madison', year=2021):
    """
    This function creates the satellite data for the given year and location
    # Usage : create_satellite_data(location = 'Madison',year=2021,urban_data=False)
    """
    sat.create_station_file(location, year, urban_data=False)
    sat.create_station_file(location, year, urban_data=True,)


def process_raster_data(clean_data, create=False, year=2021, location='Madison'):
    """
    This function processes the raster data for the given year and location
    Cleaning the rows and renaming
    """
    if create:
        create_satellite_data(location, year)

    # This data is reads the tiff files created by 5. Combining Ecostress.ipynn
    ecostress_data = pd.read_csv(TRAIN_ECOSTRSS_FILE)
    urban_data = pd.read_csv(TRAIN_URBAN_FILE)

    ecostress_data = ecostress_data[[
        'station', 'latitude', 'longitude', 'value_LST', 'hour']]
    urban_data = (urban_data.iloc[:, 1:]).drop(
        columns=['beg_time', 'geometry'])
    urban_data['valueBuildingheight'] = urban_data['valueBuildingheight'].fillna(
        0)

    # result_df = sat.station_daily_lst_anomaly_means()
    result_df = ecostress_data[['station', 'hour', 'value_LST']]
    result_df = result_df.rename(columns={'value_LST': 'adjusted_lst'})
    result_df.station = result_df.station.str.upper()

    updated_data = pd.merge(clean_data, result_df, on=[
                            'station', 'hour'], how='left')
    updated_data = pd.merge(updated_data, urban_data, on=[
                            'station', 'latitude', 'longitude'], how='inner')

    return updated_data


def null_fill_strategy(final_df, strategy='mean'):
    ''' This function fills the null values in the final_df with the given strategy
    '''
    if strategy == 'mean':
        list_of_cols = [x for x in final_df.columns if 'closest_station' in x]
        for x in list_of_cols:
            final_df[x] = final_df[x].interpolate(
                method='linear', limit_direction='both')

    return final_df


def get_final_df(new_data, updated_data, spatial_columns):
    ''' new_data : consists of closest stations and temperatures
        updated_data : consists of all the satellite data
    '''
    new_data = new_data[spatial_columns]
    final_df = pd.merge(new_data, updated_data, on=[
                        'station', 'beg_time', 'temperature', 'latitude', 'longitude'], how='inner')

    final_df = null_fill_strategy(final_df, strategy='mean')

    return final_df


def split_(sequence, window_size):
    ''' sequence : input array of tempearture values : num_sample * 1
        window_size : number of lagged values to be used as features
    '''

    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + window_size
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        seq_y = sequence.iloc[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def get_time_adjusted_df(final_df, start_end_date=None, window_size=5, column='temperature'):
    ''' This function creates the time adjusted dataframe with lagged values
    '''
    # window_size            #480 = 24 obs per day * 20 days : data from past 20 days is taken as features
    # number_of_days = 720            #720 = 24 obs per day * 30 days : to predict for next 30 days

    if start_end_date is None:
        start_date = 70
        end_date = 180
    
    else:
        start_date = start_end_date[0]
        end_date = start_end_date[1]


    # final_df_slice = final_df.query(f'day_of_year > {start_date} and day_of_year < {end_date}')
    final_df_slice = final_df
    series = final_df_slice[column]

    x_train, y_train = split_(series, window_size)

    columns = ['t_'+str(i) for i in range(window_size, 0, -1)]
    final_df_ = final_df_slice[window_size:].reset_index(drop=True)
    temp_ = pd.DataFrame(x_train, columns=columns)
    time_adjusted_df = pd.concat([final_df_, temp_], axis=1)
    time_adjusted_df.sort_values(['station', 'beg_time'], inplace=True)

    # 1+ because distance can be zero
    time_adjusted_df['closest_1_distance'] = 1 / \
        (1+(time_adjusted_df['closest_1_distance']))
    time_adjusted_df['closest_2_distance'] = 1 / \
        (1 + (time_adjusted_df['closest_2_distance']))
    time_adjusted_df['closest_3_distance'] = 1 / \
        (1 + (time_adjusted_df['closest_3_distance']))
    time_adjusted_df['valueNearestDistWater'] = 1 / \
        (1 + (time_adjusted_df['valueNearestDistWater']))

    return time_adjusted_df


def get_train_test_data(final_df, window_size=5):
    ''' This function creates the train and test data from time adjusted dataframe'''

    time_adjusted_df = get_time_adjusted_df(final_df, start_end_date=None, window_size=window_size, column='temperature')
    
    if 'beg_time' in time_adjusted_df.columns:
        available_columns = ['station', 'temperature','beg_time']
    else:
        available_columns = ['station', 'temperature']

    X = time_adjusted_df.drop(available_columns, axis=1)
    
    y = time_adjusted_df['temperature']

    # will be required later on to find lagged values in prediction
    time_adjusted_df.to_csv('Analytics/time_adjusted_df.csv', index=False)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, shuffle=False)

    # will be required later on to align columns between train and test
    # x_test.head().to_csv('Analytics/X_test.csv', index=False)
    return x_train, x_test, y_train, y_test


def plot_op(model, x_test, y_test, hour_filter):
    ''' Plots the predicted and true values : Used by train save function'''

    # plot_df = x_test.copy()
    # predictions = model.predict(plot_df)
    # plot_df['predicted_temperature'] = predictions
    # plot_df['true_temperature'] = y_test

    # if hour_filter is False:
    #     plot_df['hour'] = 25

    # # plt.plot(plot_df['predicted_temperature'],
    # #          label='predicted', color='orange')
    # # plt.plot(plot_df['true_temperature'], label='true')
    # # plt.legend()

    # error_score = metrics.mean_squared_error(y_test,predictions,squared=False,)
    # print(f'Root Mean Squared Error : {error_score}')
    # plt.show()

    # cols_of_interest = ['true_temperature', 'predicted_temperature', 'hour']
    # if 'latitude' in plot_df.columns:
    #     cols_of_interest = ['latitude', 'longitude',
    #                         'true_temperature', 'predicted_temperature', 'hour']

    # op_frame = plot_df[cols_of_interest]

    # # return op_frame
    # if len(op_frame['hour'].unique()) > 1:
        
    #     # If multiple hours are present, then we do a line plot
    #     op_frame.groupby(['hour']).mean()[
    #         ['true_temperature', 'predicted_temperature']].plot()
    #     plt.title('True vs Predicted Temperature for 24 hours')
    #     plt.ylabel('Temperature in degree Celsius')

    # else:
    #     # If only one hour is present, then we do a bar plot
    #     true_temp = op_frame['true_temperature'].mean()
    #     predicted_temp = op_frame['predicted_temperature'].mean()
    #     # plt.bar(['true'], [true_temp,], color=[
    #     #         'orange'], label='true', alpha=0.5)
    #     # plt.bar(['predicted'], [predicted_temp,], color=[
    #     #         'blue'], label='predicted', alpha=0.5)
    #     # plt.legend()
    #     # plt.show()


    # return plot_df,error_score

    plot_df = x_test.copy()
    plot_df['predicted_temperature'] = model.predict(plot_df)
    plot_df['true_temperature'] = y_test
    if len(hour_filter) == 1:
        plot_df['hour'] = hour_filter[0]
    error_score = metrics.mean_squared_error(plot_df['true_temperature'],plot_df['predicted_temperature'],squared=False,)

    return plot_df,error_score

    


def train_save(modelx, data,hour_filter,neural_net=None,clean_directory = False,fit=True):
    ''' Function to train and save the model
        modelx : model object to be trained 
        model_name : name of the model
        neural_net : A dictionary thats passed to the model.fit function if neural net
                    argyments : epochs, batch_size, verbose = 2
      '''
    # last_file = sorted([ x.split('.')[0][-1] for x in os.listdir(MODEL_PATH) if model_name in x ])[-1]
    model_name = modelx.__class__.__name__
    model_path = os.path.join(MODEL_PATH,model_name)
    os.makedirs(model_path, exist_ok=True)
    
    d_train, d_test, y_train, y_test = data[0], data[1], data[2], data[3]
    
    # To save up the list of columns used in the model for prediction (closest temps)
    # d_test.head().to_csv('Analytics/X_test.csv', index=False)

    #To save up the list of columns used in the model for prediction (closest temps)
    cols_list = data[0].head()

    #when its bulk mode, we dont want to clean the directory
    #make sure first file is not deleted
    if clean_directory:
        # print('Cleaning')
        helper.clean_directory(model_path)
    
    cols_list.to_csv(os.path.join(model_path,model_name+'_cols_list.csv'), index=False)

    # To save up the model
    temp = os.listdir(model_path)
    temp = [x for x in temp if model_name in x and 'cols_list' not in x]
    if len(temp) == 0:
        new_file = 0
    else:

        last_file = sorted([int(x.split('.')[0][len(model_name)+1:])
                       for x in os.listdir(model_path) if model_name in x and 'cols_list' not in x])[-1]
        new_file = int(last_file)+1
    file_name = f'{model_path}/{model_name}_{new_file}.sav'

    # train the model
    if fit is True:
        if neural_net is not None:
            modelx.fit(d_train, y_train, epochs=neural_net['epochs'], batch_size=neural_net['batch_size'], verbose=1)
        else: 
            modelx.fit(d_train, y_train)

    elif fit is False:
        print('Skipping model fit')
        


    # save the model using pickle
    print(f'Model Saved : {file_name}')
    pickle.dump(modelx, open(file_name, 'wb'))

    predictions_df,error_score = plot_op(modelx, d_test, y_test,hour_filter)
    return predictions_df,error_score


