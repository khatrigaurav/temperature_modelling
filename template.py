import os
import sys
import shutil


args = sys.argv[1:]

try:
    lat_1 = args[1]
    lat_2 = args[2]
    lon_1 = args[3]
    lon_2 = args[4]
    location = args[0]

except Exception as e:
    print('\nPlease provide the following arguments: location, lat_1, lat_2, lon_1, lon_2')
    print('Usage: python template.py <location> <lat_1> <lat_2> <lon_1> <lon_2> \n')
    exit(1)

# try:
#     os.makedirs(f'{location}', )

# except Exception as e:
#     print(f'{location} already exists')
#     exit(1)


# Replace 'source_dir' and 'destination_dir' with your desired directory paths
SOURCE_DIR      = 'Template/'
location        = location[0].upper()+location[1:]
destination_dir = os.path.join('Codebase',location)

try:
    shutil.copytree(SOURCE_DIR, destination_dir,)

    print(f"Directory '{SOURCE_DIR}' copied to '{destination_dir}' successfully.")

    #replace the location name and latitude in the config file
    with open(os.path.join(destination_dir,'wunder_config.ini'), 'r') as file :
        filedata = file.read()
    # Replace the target string
    filedata = filedata.replace('location_filler', location)
    filedata = filedata.replace('lat_1_filler', lat_1)
    filedata = filedata.replace('lat_2_filler', lat_2)
    filedata = filedata.replace('lon_1_filler', lon_1)
    filedata = filedata.replace('lon_2_filler', lon_2)


    # Write the file out again
    with open(os.path.join(destination_dir,'wunder_config.ini'), 'w') as file:
        file.write(filedata)


except FileExistsError:
    # shutil.copytree(SOURCE_DIR, destination_dir)
    print(f"Directory '{destination_dir}' already exists.")

