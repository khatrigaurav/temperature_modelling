# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:25:59 2023

@author: tvo

Script to extract values of underlying raster from the points

Step 1:
    
    Reproject the shapefile of polypoints to the same projections
    with raster images of ecostress and nlcd
    
    
    
Step 2:
    
    Using function extract values from points to get the values of
    either surface properties or LST values from ecostress imaginary
    
    Note: make sure the return datetime from ecostress is in LOCAL hours.
    
    
    
    
"""

# ****** Required packages ************************#
import geopandas as gpd
import rioxarray as rxr
import pandas as pd
from shapely.geometry import Point
from pyproj import CRS
import rasterio
import os
import datetime
from functools import reduce
import numpy as np
# ****** Required packages ************************#

# Homing : Setting correct path
SAVE_DIRECTORY = "/Users/gaurav/UAH/temperature_modelling/data/raster_op/"


def reproject(file_raster, file_csv):
    """
    Function to reproject the shapefile to a desired coordinate

    Example:

        file_raster = str('//uahdata/rgroup/urbanrs/'+
                         'TRANG_DATA/NASA_HAQUAST/'+
                         'final_folder_combine/Madison/ECOSTRESS/'+
                         'geotiff_clipped_stateplane/'+
                         'ECO2LSTE.001_SDS_LST_doy2018259095525_aid0001_clipped_stateplane.tif'
                         )

        file_csv = str('//uahdata/rgroup/urbanrs/'+
                       'TRANG_DATA/NASA_HAQUAST/station_list.csv')



        df_station_reproj = reproject(file_raster,file_csv)        
    """
    # read an example of raster file with the coordinate
    rs = rxr.open_rasterio(file_raster)
    des_crs = rs.spatial_ref.crs_wkt

    # read .csv file with coordinates (by default in WGS84)
    df_station = pd.read_csv(file_csv)

    # Transform from .csv to geopandas using latitude and longitude coor
    df_station = gpd.GeoDataFrame(df_station)
    df_station['geometry'] = df_station.apply(lambda row:
                                              Point(row.longitude, row.latitude), axis=1)
    crs_wgs84 = CRS.from_epsg(4326)  # epsg code for WGS84
    df_station.crs = crs_wgs84

    # rerpject to the desired coodinate
    df_station_reproj = df_station.to_crs(des_crs)

    return df_station_reproj


def extract_raster_values(df_station_reproj, folder_raster, local_time_zone):
    '''
    Function to extract the values of raster for each points 
    of the reprojected shapefile
    Example:
        df_station_reproj = reproject(file_raster,file_csv)
        # convert to local time
        # check here for list of local time zone naame
        # https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        # column 'TZ indentifier'
        # for example, MAdison, WI has 'America/Chicago' (CST time zone)
        local_time_zone = tz.gettz('America/Chicago')
        folder_raster = str('//uahdata/rgroup/urbanrs/TRANG_DATA/'+
                            'NASA_HAQUAST/final_folder_combine/'+
                            'Madison/ECOSTRESS/geotiff_clipped_stateplane/')

        df_output = extract_raster_values(df_station_reproj, folder_raster, local_time_zone)
    '''
    # create a empty list to hold the results #
    output_list = list()
    column_name_list = list()

    for file_raster in os.listdir(folder_raster):

        if 'geotiff_clipped_stateplane' not in folder_raster:
            df_station_reproj_copy = df_station_reproj.copy()

        # print(file_raster)
        if 'geotiff_clipped_stateplane' in folder_raster:
            column_name = 'value_LST'

        else:
            column_name = 'value'+file_raster.split('_')[0]

        # read the raster
        rs = rasterio.open(folder_raster+file_raster)

        # read the first layer of the raster file
        band = rs.read(1)

        # create an empty list to store the raster values output
        value_list = list()

        if 'geotiff_clipped_stateplane' in folder_raster:
            df_station_reproj_copy = df_station_reproj.copy()

        for point in df_station_reproj_copy['geometry']:

            x = point.xy[0][0]
            y = point.xy[1][0]

            # get indexes of row, col that the point falling over
            # the raster layer
            row, col = rs.index(x, y)

            # return in the raster values from the row,col indexes
            value = band[row, col]

            # for building height, there are missing values
            # replace with nan
            if 'height' in file_raster:
                if value < 0:
                    value = np.nan

            # append the value to the list
            value_list.append(value)

        # create a new column from the new list of raster values
        # and add to the existing dataframe

        df_station_reproj_copy[column_name] = value_list

        # read the file name of the ecostress file and assign the local time
        if 'geotiff_clipped_stateplane' in folder_raster:
            # read time in UTC

            time_utc = file_raster.split('_')[3].split('doy')[-1]

            print(time_utc)
            local_time = datetime.datetime.strptime(time_utc, '%Y%j%H%M%S')

            local_time = local_time.replace(tzinfo=local_time_zone)

            print(local_time)
            df_station_reproj_copy['localtime'] = [
                local_time]*len(df_station_reproj_copy)
            df_station_reproj_copy['filename'] = [
                file_raster]*len(df_station_reproj_copy)

        output_list.append(df_station_reproj_copy)
        column_name_list.append(column_name)

    # concat the outputlist to a new output dataframe
    if 'geotiff_clipped_stateplane' not in folder_raster:

        df_output = reduce(lambda x, y: pd.merge(x, y,
                                                 left_index=True,
                                                 right_index=True),
                           output_list)

    else:
        df_output = pd.concat(output_list)

    # export to .csv file

    if 'geotiff_clipped_stateplane' in folder_raster:
        column_names = ['station', 'latitude', 'longitude','beg_time', 'geometry']+['value_LST',
                                                                         'localtime',
                                                                         'filename']

        df_output = df_output[column_names]
        save_loc = os.path.join(SAVE_DIRECTORY, 'ECOSTRESS_values.csv')
        # print(save_loc)
        df_output.to_csv(save_loc)

    else:
        column_names = ['station', 'latitude',
                        'longitude','beg_time', 'geometry']+column_name_list

        df_output = df_output[column_names]
        save_loc = os.path.join(SAVE_DIRECTORY, 'urban_surface_properties_values.csv')
        df_output.to_csv(save_loc)

    return 
