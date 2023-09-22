"""
Created on March1


@author: Gaurav
@version : 1.1

@Revision1 : March1

This script saves all the weather stations in a given lat-long range for Orlando area

Usage: python orlando_list_weather_stations.py <pws/observation/airport>
Source : WUNDERGROUND
"""

import itertools
import sys
import requests
import numpy as np
import pandas as pd
import os
import configparser


#Section to read variables from config file
config = configparser.ConfigParser()
config.read('wunder_config.ini')
API_KEY = config['WUNDERGROUND']['API_KEY']
LOCATION_NAME = config['WUNDERGROUND']['LOCATION_NAME']
LONG1 = float(config['WUNDERGROUND']['LONG1'])
LONG2 = float(config['WUNDERGROUND']['LONG2'])
LAT1 = float(config['WUNDERGROUND']['LAT1'])
LAT2 = float(config['WUNDERGROUND']['LAT2'])

FILEPATH = '../data/'+LOCATION_NAME+'/'

if len(sys.argv)<2:
    print("Usage: python orlando_list_weather_stations.py <pws/observation/airport>")
    sys.exit(1)

ITEM = sys.argv[1]

if not os.path.exists(FILEPATH):
    print(FILEPATH)
    os.mkdir(FILEPATH)

OUTPUT_FILENAME = FILEPATH +ITEM + "_station_lists_" + LOCATION_NAME.lower()+ ".csv"

lat_range = np.linspace(LAT1,LAT2,num=15)
lon_range = np.linspace(LONG1,LONG2,num=15)

entire_range = [k for k in itertools.product(lat_range,lon_range)]

df = pd.DataFrame(columns=["stationId","latitude","longitude","products"])
print(len(entire_range))

for location_map in entire_range:
    lat = location_map[0]
    lon = location_map[1]
    # print(f'Finding weather station lists for {ITEM}')
    CMD = 'https://api.weather.com/v3/location/near?geocode='+str(lat)+","+str(lon)+'&product='+ITEM+'&format=json&apiKey='+API_KEY
    # print(CMD)
    try:
        data = requests.get(CMD,timeout=4).json()
        data_dic = {'stationId':data['location']['stationId'],'latitude':data['location']['latitude'],
                    'longitude':data['location']['longitude'], 'products':ITEM}
        df = pd.concat([df,pd.DataFrame(data_dic)],ignore_index=True).drop_duplicates()
        print(f'{location_map} Completed')
    
    except Exception as e:
        print("Exception encountered , {}", e)
        print(data)

df.to_csv(OUTPUT_FILENAME,index=False)
