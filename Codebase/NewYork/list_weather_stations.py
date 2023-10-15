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

#path setup
os.chdir('../')
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

# entire_range = [k for k in itertools.product(lat_range,lon_range)]
entire_range = [(40.7568572,-73.7562042999999), (40.6883291,-73.9714049999999), (40.6097282,-73.9483038999999), (40.8448633,-73.8295406999999), 
                (40.7090573,-73.9425488999999), (40.862012,-73.9210376999999), (40.8175038,-73.9474936999999), (40.6344192,-73.8873662999999),
                (40.7877898,-73.9384126999999), (40.6853102,-73.7829680999999), (40.5975233,-73.7736057999999), (40.7730609,-73.9328955999999), 
                (40.5721526,-73.9911466999999), (40.6486826,-73.9530559999999), (40.5929243,-73.9379308999999), (40.6697325,-73.9617621999999), 
                (40.7576102,-73.9081958999999), (40.6349633,-73.9816963999999), (40.750627,-73.8287364999999), (40.6915191,-74.0202531999999), 
                (40.5979556,-74.1246275999999)] 

df = pd.DataFrame(columns=["stationId","latitude","longitude","products"])
# print(len(entire_range))

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
