"""
Created on : Nov 18, 2023

@author: Gaurav
@version: 1.0

Functions to help in plotting outputs

Functions:

"""

import plotly.express as pe
import plotly.offline as pyo
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import rioxarray as rxr
import pandas as pd
import os
from lib.helper import resample_daily
from sklearn import metrics
# Constants and variable paths for raster plots

SHAPEFILE_PATH = '/Users/gaurav/UAH/ECOSTRESS_and_urban_surface_dataset/Madison/shpfile/Madison_WI_UA_mer.shp'
BOUNDARY_GDF = gpd.read_file(SHAPEFILE_PATH)
TARGET_CRS = BOUNDARY_GDF.crs
BOUNDARY_GDF_EPS = BOUNDARY_GDF.to_crs(epsg=6879)
SAVE_PATH = '/Users/gaurav/UAH/temperature_modelling/Resources/temperature_plots/Madison'
# scatter_data_directory = '/Users/gaurav/UAH/temperature_modelling/Analytics/temp'

path_ = '/Users/gaurav/UAH/ECOSTRESS_and_urban_surface_dataset/Madison/ECOSTRESS/geotiff_clipped_stateplane/ECO2LSTE.001_SDS_LST_doy2022167193705_aid0001_clipped_stateplane.tif'

temperature_data = rxr.open_rasterio(path_)


def plot_(dfx, animation_frame_comp='day_of_year', frame_duration=200, station_name=None, resample=True):
    '''
    dfx                     : Source dataframe
    animation_frame_comp    : defines which column to animate upon (eg. year, month)
    frame_duration          : defines how fast the animation should be
    station_name            : Optional argument to plot a single station

    Usage                   : dp.plot_(df_vegas,'day_of_year',180)
    '''
    # fig = pe.scatter_mapbox(dfx,lat='latitude',lon = 'longitude',animation_frame='day_of_year',color = 'temperature',range_color=[-40,40],hover_data=['station'],height = 610)
    # Making the legend dynamic

    if resample:
        dfx = resample_daily(dfx)

    if station_name:
        dfx = dfx[dfx.station == station_name]

    dfx = dfx.sort_values(by=[animation_frame_comp])
    fig = pe.scatter_mapbox(
        dfx, lat='latitude',
        lon='longitude',
        animation_frame=animation_frame_comp,
        color='temperature',
        hover_data=['station'],
        height=710,
        color_continuous_scale='thermal',
        # width=200
    )

    fig.update_layout(mapbox_style='open-street-map',)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_traces(marker=dict(size=14, color='black'))
    # fig.update_layouta(updatemenus=dict())

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = frame_duration
    # fig.layout.coloraxis.colorbar.title.text = 'Temperature - °C'

    # fig.show()
    return fig


def get_raster(df_, temp_col='prediction_temps', pixel_size=150):
    ''' This function takes in a dataframe of format : lat , long, 
        temperature and returns a raster of the temperature values
    '''

    gdf = df_[['latitude', 'longitude', temp_col]]
    gdf['geometry'] = [Point(xy)
                       for xy in zip(gdf['longitude'], gdf['latitude'])]
    gdf = gpd.GeoDataFrame(
        gdf, geometry=gdf['geometry'], crs=temperature_data.rio.crs)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    rows = int((ymax - ymin) / pixel_size)
    cols = int((xmax - xmin) / pixel_size)

    # Create the raster
    temperature_raster = np.zeros((rows, cols), dtype='float32')

    # Assign temperature values to the raster cells
    for index, row in gdf.iterrows():

        col = int((row['longitude'] - xmin) / pixel_size)
        r = int((ymax - row['latitude']) / pixel_size)

        # Check if indices are within bounds
        if 0 <= r < rows and 0 <= col < cols:
            temperature_raster[r, col] = row[temp_col]

    return temperature_raster, (xmin, xmax, ymin, ymax)


def get_plot(temperature_raster, bounds, cmap='coolwarm', change_null=False, plot_boundary=True,hour=None,model_name = None):
    ''' This function takes in a raster which is output of previous function 
        and plots it
        Usage : get_plot(raster1, bounds,cmap = 'coolwarm',change_null=True)
    '''
    xmin, xmax, ymin, ymax = bounds
    
    if model_name:
        op_path =os.path.join(SAVE_PATH,model_name)
        os.makedirs(op_path,exist_ok=True)

    if change_null:
        temperature_raster[temperature_raster == 0] = np.nan

    plt.figure(figsize=(10,10))

    plt.imshow(temperature_raster, extent=(
        xmin, xmax, ymin, ymax), cmap=cmap, origin='upper')

    if plot_boundary:
        BOUNDARY_GDF_EPS.boundary.plot(
            ax=plt.gca(), edgecolor='black', linewidth=1)

    # gdf.plot(ax=plt.gca(), color='red', markersize=1)
    plt.colorbar(label='Temperature (C)', shrink=0.5)
    

    data = pd.read_csv('/Users/gaurav/UAH/temperature_modelling/Analytics/temp_data/grouped_data.csv')
    if hour is not None:
        data = data[data.hour == hour]
        plt.scatter(data.longitude,data.latitude,c=data.temperature,s=50,cmap='coolwarm',edgecolors='black',linewidths=1)
        plt.title(f'Predicted Temperature for hour {hour}')
        plt.savefig(f'{op_path}/scatter_hour_{str(hour).zfill(2)}.jpeg',dpi=300)
        print(f'Saved scatter plot in {op_path}')


#coolwarm
def get_subplots(raster1,raster2,bounds,hour_filter=None,cmap='coolwarm',save=False,model_name=None):
    ''' This function returns a figure with two subplots
        compared with adjusted LST temperature
    '''

    plt.figure(figsize=(18, 16))

    if model_name:
        op_path =os.path.join(SAVE_PATH,model_name)
        os.makedirs(op_path,exist_ok=True)

    # Plot the first subplot
    plt.subplot(1, 2, 1)
    get_plot(raster1, bounds,cmap = cmap,change_null=True)
    plt.title('Predicted Temperature')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Plot the second subplot
    plt.subplot(1, 2, 2)
    get_plot(raster2, bounds,cmap = cmap,change_null=True)
    plt.title('Adjusted LST Temperature ')
    plt.xlabel('Longitude')
    # plt.ylabel('Latitude')


    # Adjust layout for better spacing
    # plt.tight_layout()

    # Show the plot
    if hour_filter:
        hour = hour_filter[0]
    else:
        hour = 'all'

    plt.suptitle(f'Predicted Temperature VS adjusted LST Temperature for hour {hour}',fontsize=15,position=(0.5,0.28))

    if save:
        plt.savefig(f'{op_path}/hour_{hour}.jpeg')

    plt.show()


def plot_feature_importances(test_column_list,model=None,hr=None,bulk_importances=None):
    ''' This function plots the feature importances of the model for RF and XGB
        Used in predictor.ipynb
    '''
    if bulk_importances is not None:
        importances = bulk_importances
        # print(importances)
    else:
        importances = model.feature_importances_
    indices = np.argsort(importances)
    # features = test_data[COL_LIST].columns.to_list()
    features = test_column_list
    plt.figure()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)),
             importances[indices], color='b', align='center')
    # plt.barh(rrf.feature_names_in_, rrf.feature_importances_)
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    if hr:
        plt.title(f'Feature Importances for hour {hr}')
    plt.show()


def plot_mean(model_dict,model_name='Random Forest',col_list=None):
    ''' This function plots the mean RMSE and mean predicted and actual temperature
        Used in Modeller.ipynb
        model_dict : Dictionary of format {hour : [prediction_data,error_value,feature_importances]}
        '''
    hours = []
    mean_error = []
    predicted = []
    actual = []
    feature_importances = []
    
    if len(model_dict.keys()) == 1:
        bulk_mode = False
        #this means that model is only trained for one hour
    else:
        bulk_mode = True

    if bulk_mode:
        for hour_ in model_dict.keys():
            hours.append(hour_)
            mean_error.append(model_dict[hour_][1].mean())
            predicted.append(model_dict[hour_][0].predicted_temperature.mean())
            actual.append(model_dict[hour_][0].true_temperature.mean())
            feature_importances.append(model_dict[hour_][2])

    if not bulk_mode:
        data = model_dict[25][0]
        for hour in sorted(data.hour.unique()):
            hours.append(hour)
            data_slice = data[data.hour == hour]
            error_val = metrics.mean_squared_error(data_slice.predicted_temperature,data_slice.true_temperature,squared=False)
            mean_error.append(error_val)
            predicted.append(data_slice.predicted_temperature.mean())
            actual.append(data_slice.true_temperature.mean())
            feature_importances.append(model_dict[25][2])


    # return hours,mean_error,predicted,actual,feature_importances
    hours = [x+1 for x in hours]
    plt.figure()
    plt.plot(hours,mean_error)
    plt.xlabel('Hour')
    plt.ylabel('RMSE')
    plt.title('Mean RMSE vs Hour for '+model_name)
    
    #plot a horizonal mean line
    plt.axhline(np.mean(mean_error), color='r', linestyle='--',label='Mean RMSE')
    plt.legend()

    plt.figure()
    plt.plot(hours,predicted)
    plt.plot(hours,actual)
    plt.xlabel('Hour')
    plt.ylabel('Temperature in C')
    plt.title('Predicted vs Actual Temperature for '+model_name)
    plt.legend(['Predicted','Actual'])

    try:
        feature_importances = np.sum(feature_importances,axis=0)
        # print(f'Feature importances for {model_name} : {feature_importances}')
        # print(f'col_list : {col_list}')
        plot_feature_importances(test_column_list=col_list,bulk_importances=feature_importances)

    except Exception as e:
        # print(e)
        print('Feature importances not available')
