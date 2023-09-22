"""
Created on May 11,2023
@author: Gaurav
@version : 1.0

@Revision1 : March1

Downloads Netatmos data for Madison.
Usage: python net_atmo.py.

Source : NETATMOS
"""

import os
import requests
import pandas as pd
import time
import datetime
import configparser

config = configparser.ConfigParser()
config.read('netatmos_config.ini')

#Authentication -- Needs to be updated every hour :(
# BEARER_KEY = config['NETATMOS']['BEARER_KEY']

START_DATE_ = config['NETATMOS']['START_DATE']  #needs to be in linux time, so will be converted later
END_DATE_ = config['NETATMOS']['END_DATE']     #1677715199 : needs to be in linux time
LAT_NE = float(config['NETATMOS']['LAT_NE'])
LON_NE = float(config['NETATMOS']['LON_NE'])
LAT_SW = float(config['NETATMOS']['LAT_SW'])
LON_SW = float(config['NETATMOS']['LON_SW'])
REQUIRED_DATA = config['NETATMOS']['REQUIRED_DATA']
FILTER = config['NETATMOS']['FILTER']
LOCATION_NAME = config['NETATMOS']['LOCATION_NAME']
CLIENT_ID = config['NETATMOS']['CLIENT_ID']
CLIENT_SECRET = config['NETATMOS']['CLIENT_SECRET']

#module to get authentication token
def get_netatmo_bearer_token(refresh_token=None):
    """Get a bearer token for the Netatmo API"""
    url = "https://api.netatmo.com/oauth2/token"
    
    if refresh_token:
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }

    else:
        payload = {
            "grant_type": "password",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "username": "gaurav113.gk@gmail.com",
            "password": "T@sr.!s9T5JntBK",
        }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        access_token =  response.json()["access_token"]
        refresh_token = response.json()["refresh_token"]
        expires_in = response.json()["expires_in"]
        expiration_time = time.time() + expires_in
        return access_token,refresh_token,expiration_time
    else:
        print(response)
        raise Exception("Failed to get bearer token")
    
BEARER_KEY,REFRESH_KEY, expiration_time  = get_netatmo_bearer_token()

#Filepaths for storing data
year = START_DATE_.split('-')[0]  #year for storing data
FILE_PATH = os.path.join('../data',LOCATION_NAME,'netatmos_data'+ '_'+LOCATION_NAME.lower())
YEAR_PATH = os.path.join('../data',LOCATION_NAME,'netatmos_data'+ '_'+LOCATION_NAME.lower(),year)

if not os.path.exists(YEAR_PATH):
    os.makedirs(YEAR_PATH, exist_ok=True)

#module to get station_lists, can be later added to config file
headers_device = {
    'accept': 'application/json',
    'Authorization': 'Bearer ' + BEARER_KEY
}

params_device = {
    'lat_ne': LAT_NE,
    'lon_ne': LON_NE,
    'lat_sw': LAT_SW,
    'lon_sw': LON_SW,
    'required_data': REQUIRED_DATA,
    'filter': FILTER
}

try:
    response_device = requests.get('https://api.netatmo.com/api/getpublicdata',
                               params=params_device, headers=headers_device,timeout=30).json()
    if response_device.get('error'):
        print('Error in getting device data')
        print(response_device)
        print(response_device.get('error'))
        exit()
        
except Exception as e:
    print('Exception in getting device data')
    print(e)



    

def get_module_ids(response):
    ''' Get the module ids for the stations'''
    body = response.get('body')
    device_data = pd.DataFrame(body)
#     device_data['module_id'] = [k if 'NAModule1' in v else 'NA' 
# for k,v in value.items() for value in device_data['module_types'].values]
    device_data['module_id'] = 'NA'
    device_data['module_type'] = 'NA'
    device_data['module_id'] = [k for value in device_data['module_types'].values
                                for k,v in value.items() if 'NAModule1' in v ]
    device_data['module_type'] = [v for value in device_data['module_types'].values
                                  for k,v in value.items() if 'NAModule1' in v ]
    device_data = device_data[device_data.module_type == 'NAModule1']
    device_data['device_info'] = device_data['_id']+','+ device_data['module_id']

    return device_data



def get_lat_lon(df,idx):
    ''' Get the lat and lon for the station using the device identifier dataframe above'''
    row_of_interest = df.loc[idx]['place']['location']
    print(f'row_of_interest: {row_of_interest}')
    return row_of_interest


def get_data(start_date,end_date,device_identifier,BEARER_KEY):
    ''' start_date : 1609459200
        This function will be ran in loops'''
    
    headers = {
        'Authorization': 'Bearer ' + BEARER_KEY,
        'Content-Type': 'application/json;charset=UTF-8',
    }
    json_data = {
        'date_begin': start_date,
        'date_end': end_date,
        'scale': 'max',
        'device_id': device_identifier[0],
        'module_id': device_identifier[1],
        'type': 'Temperature,Humidity',
    }

    try:
        response = requests.post('https://app.netatmo.net/api/getmeasure', headers=headers, json=json_data,timeout=30)
    
        return response.json()
    
    except Exception as exception:
        print('Exception in getting data')
        print(exception)

def process_api_response(z,start_date,end_date):
    
    '''I/p is op of get_data() function'''
    temp = pd.DataFrame(z)
    
    if len(temp) ==0:
        print(f'No data for given duration {start_date},{end_date}')
#         print(z)
#     print(f'Length of elements {len(temp)}')
    
    else:
        temp['value'] = temp['value'].apply(lambda x: max(x))
        temp['temperature'] = temp['value'].apply(lambda x : x[0])
        temp['humidity'] = temp['value'].apply(lambda x : x[1])
        temp['beg_time'] = temp['beg_time'].apply(lambda x : (datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")))
        temp = temp.drop(['value','step_time'],axis=1)
    
        return temp


def combine_data():
    
    '''Final function to combine them all'''
    BEARER_KEY,REFRESH_KEY, expiration_time  = get_netatmo_bearer_token()
    
    # Convert datetime object to epoch time
    start_date = int(datetime.datetime.strptime(START_DATE_, "%Y-%m-%d").timestamp())
    end_date = int(datetime.datetime.strptime(END_DATE_, "%Y-%m-%d").timestamp())
    
    # counter_date = 308100
    starting_df = pd.DataFrame(columns=['beg_time','temperature','humidity'])
    time1 = time.time()
    df_device_info = get_module_ids(response_device)
    station_list = [list(k) for k in [x.split(',') for x in df_device_info.device_info]]

    
    #Error module : New code to start from where it left off 
    if "temp_file.txt" not in os.listdir():
        open("temp_file.txt",'w+').write('0')
    
    idx_new = open("temp_file.txt",'r+').read()
    station_list = station_list[int(idx_new):]


    for idx,element in enumerate(station_list):
        lat_long = get_lat_lon(df_device_info,idx)
        begin_date = start_date
        count = 308100
        starting_df = pd.DataFrame(columns=['beg_time','temperature','humidity'])
        
        csv_name = 'Station_'+str(idx+int(idx_new)+1)+'.csv'
        print(f'Processing data for {csv_name}')
        
        while begin_date+count < end_date:
            
            if int(time.time()) >= expiration_time:
                BEARER_KEY,REFRESH_KEY, expiration_time  = get_netatmo_bearer_token(REFRESH_KEY)

            data = get_data(str(begin_date),str(begin_date+count),element,BEARER_KEY)
            data = data.get('body')
            processed_response = process_api_response(data,begin_date,end_date)
            begin_date = begin_date+count
            
            starting_df = pd.concat([starting_df,processed_response])
        
        # starting_df['latitude'] = lat_long.get('lat')
        # starting_df['longitude'] = lat_long.get('lon')
        starting_df['latitude'] = lat_long[1]
        starting_df['longitude'] = lat_long[0]
        
        file_name = os.path.join(YEAR_PATH,csv_name)
        starting_df.to_csv(file_name)
        
        print(f'Time_taken: {time.time()-time1} seconds ')
        print('Sleeping for 15 seconds')
        time.sleep(3)
        
        with open('temp_file.txt','w') as f:
            f.write(str(idx+int(idx_new)))
            f.close()
    

combine_data()
os.remove('temp_file.txt')

