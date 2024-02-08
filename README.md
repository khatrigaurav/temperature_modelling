Data Extraction Pipeline:

1) Wunderground data <br>
    a) PWS/Observation Stations/Airport<br>
        1) Get the station lists : <br>
            1) Update wunder_config.ini for API_KEY and lat long ranges.<br>
            2) Go to /Orlando/ and run "python list_weather_stations.py [pws/observation/airport]". It will save corresponding file in  data/Orlando/. <br>

            This saves a list of stations in data/Orlando/<pws>_station_lists_<orlando>.csv.<br>


        2) Download the corresponding data: <br>
            Scripts used : pws_script.py for pws data, observation_script.py for observation_data and parallel.py.
            Usage : 
    
            2.1) python parallel.py pws 
            This uses the pws_script.py , which collects data on a daily basis and hence takes time.

            2.2) python parallel_obs.py, 
                which collects data from observation_script.py in monthly basis and is hence faster.

    Known Issues : You may need to go to wunderground and generate new "API KEY" time and again

    b) 
    
    
2) Netatmos Data

3) LTER Data
