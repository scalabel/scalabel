import argparse
from pathlib import Path
import csv
import numpy as np
import json
import os
import hdf5plugin
from project_pointcloud import projectPoints
from event_preparer import EventPreparer

EVENT_TIME_DELTA_US = 15000
EVENT_HEIGHT = 720
EVENT_WIDTH = 1280

"""
Process Step 1: Preprocessing the data

The script generates and groups overlays to support the labeling
of the main rgb image. Further, it generated the configu-
ration files necessary to open a new project with this data
in Scalabel. 

Please note that this script is intended to be modified to the
specific needs of the user.
"""

def timestamps_reader(timestamps_path):
    timestamps = []
    fp = open(timestamps_path, 'r')
    rdr = csv.reader(filter(lambda row: row[0]!='#',fp))
    for row in rdr:
        timestamps.append(row)
    timestamps = np.array(timestamps,np.int64)
    return timestamps


def list_full_paths(directory):
    return sorted([Path(os.path.join(directory, file)) for file in os.listdir(directory)])

def pre_process_lidar(lidar_folder_path):
    lidar_list = list_full_paths(lidar_folder_path)
    lidar_projections_path = lidar_folder_path.parent / "lidar_img"
    
    if not os.path.isdir(lidar_projections_path):
        os.makedirs(lidar_projections_path)
    
    projectPoints(lidar_list, lidar_projections_path)
    return lidar_projections_path

def pre_process_events(h5_file_path, rgb_timestamps, event_time_delta_us):
    event_representations_path = h5_file_path.parent / "event_img"
    
    if not os.path.isdir(event_representations_path):
        os.makedirs(event_representations_path)
    
    event_prep = EventPreparer(h5_file_path, event_time_delta_us, EVENT_HEIGHT, EVENT_WIDTH)
    first_event = event_prep.get_start_time_us()
    last_event = event_prep.get_final_time_us()
    if (sum(rgb_timestamps[:,0]>=first_event)>0):
        first_index = np.argmax(rgb_timestamps[:,0]>=first_event)
    else:
        print("Error, Events do not correlate with rgb frames") 
    last_index = np.argmin(rgb_timestamps[:,0]<=last_event) #could cause error
    if last_index == 0:
        last_index = len(rgb_timestamps)-1
    event_prep.create_representations(rgb_timestamps[first_index:last_index,0], event_representations_path, 1200,1920)

    return (first_index, last_index)

# Expected folder structure:
# data_folder
#   - rgb
#   - radar
#   - lidar
#   - events.h5
#   - rgb_timestamps.txt
#   - radar_timestamps.txt
#   - lidar_timestamps.txt
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepeare data for Annotation')
    parser.add_argument('data_folder', type=str, help='Path data folder')
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    assert data_folder.is_dir(), "Directory {} does not exist".format(str(data_folder)) 

    event_time_delta_us = EVENT_TIME_DELTA_US #TODO: make parser arg

    # Check which sensors are present -----------------------------------------
    sensor_dict = {
        "rgb": False,
        "lidar": False,
        "events": False,
        "radar": False
    }

    rgb_folder_path = data_folder / "rgb"
    if not rgb_folder_path.is_dir():
        print("No rgb folder found" +"Directory {} does not exist".format(str(rgb_folder_path)))
    else:
        sensor_dict["rgb"] = True
        rgb_list = list_full_paths(rgb_folder_path)

    lidar_folder_path = data_folder / "lidar"
    if not lidar_folder_path.is_dir():
        print("No lidar folder found" +"Directory {} does not exist".format(str(lidar_folder_path)))
    else:
        sensor_dict["lidar"] = True
        lidar_list = list_full_paths(lidar_folder_path)

    h5_file_path = data_folder / "events.h5"
    if not h5_file_path.is_file():
        print("No events.h5 file found" +"File {} does not exist".format(str(h5_file_path)))
    else:
        sensor_dict["events"] = True

    radar_folder_path = data_folder / "radar/bev"
    if not radar_folder_path.is_dir():
        print("No radar folder found" +"Directory {} does not exist".format(str(radar_folder_path)))
    else:
        sensor_dict["radar"] = True
        radar_list = list_full_paths(radar_folder_path)

    print("Sensor avilability: ", sensor_dict)

    #--------------------------------------------------------------------------

    # Check if all timestamps are present -------------------------------------
    if sensor_dict["rgb"]:
        rgb_timestamps_path = data_folder / "rgb_timestamps.txt"
        assert rgb_timestamps_path.is_file(), "File {} does not exist".format(str(rgb_timestamps_path))
        rgb_timestamps = timestamps_reader(rgb_timestamps_path)

    if sensor_dict["lidar"]:
        lidar_timestamps_path = data_folder / "lidar_timestamps.txt"
        assert lidar_timestamps_path.is_file(), "File {} does not exist".format(str(lidar_timestamps_path))
        lidar_timestamps = timestamps_reader(lidar_timestamps_path)

    if sensor_dict["radar"]:
        radar_timestamps_path = data_folder / "radar_timestamps.txt"
        assert radar_timestamps_path.is_file(), "File {} does not exist".format(str(radar_timestamps_path))
        radar_timestamps = timestamps_reader(radar_timestamps_path)

    #--------------------------------------------------------------------------

    # Create Overlays ---------------------------------------------------------
    if sensor_dict["lidar"]:
        lidar_projections_path = pre_process_lidar(lidar_folder_path)
        
    if sensor_dict["events"]:
        if sensor_dict["rgb"]:
            first_index, last_index = pre_process_events(h5_file_path, rgb_timestamps, event_time_delta_us)
            event_representations_path = h5_file_path.parent / "event_img"
        else:
            print("Events cannot be processed without rgb timestamps!")
    #--------------------------------------------------------------------------

    # Create Config file ------------------------------------------------------
    if not sensor_dict["rgb"]:
        print("No rgb folder found, cannot create config file")
    else:
        
        # dont judge, super inefficent, feel free to improve if you are bored
        file_idx = np.zeros((len(rgb_timestamps), len(sensor_dict)),dtype=np.int64)
        for i in range(len(rgb_timestamps[:,0])):
            file_idx[i,0] = i
            if sensor_dict["lidar"]:
                for j in range(len(lidar_timestamps[:,0])):
                    if (rgb_timestamps[i,0] <= lidar_timestamps[j,0]):
                        file_idx[i,1] = j
                        break
            if sensor_dict["radar"]:
                for j in range(len(radar_timestamps[:,0])):
                    if (rgb_timestamps[i,0] <= radar_timestamps[j,0]):
                        file_idx[i,3] = j
                        break
        
        if sensor_dict["events"]:
            file_idx[first_index:last_index,2] = np.array(range(last_index-first_index))

        
        # make json
        frames = []
        frameGroups = []
        if not sensor_dict["events"]:
            first_index = 0
            last_index = len(rgb_timestamps[:,0])
        else:
            event_list = list_full_paths(event_representations_path)
        
        if sensor_dict["lidar"]:
            lidar_img_list = list_full_paths(lidar_projections_path)

        for i in file_idx[first_index:last_index,0]:
            frameGroups_current = []
            # rgb
            name_rgb = rgb_list[i].stem
            frames.append(
                {
                    "name": name_rgb,
                    "url": str(rgb_list[i].relative_to(*rgb_list[i].parts[:5])),
                    "timestamp": rgb_timestamps[i,0]/1000000
                })
            frameGroups_current.append(name_rgb)
            
            #lidar
            if sensor_dict["lidar"]:
                lidar_index = file_idx[i,1]
                name_lidar = lidar_img_list[lidar_index].stem
                frames.append(
                {
                    "name": name_lidar,
                    "url": str(lidar_img_list[lidar_index].relative_to(*lidar_img_list[lidar_index].parts[:5])),
                    "timestamp": rgb_timestamps[i,0]/1000000
                })
                frameGroups_current.append(name_lidar)
            #events
            if sensor_dict["events"]:
                event_index = file_idx[i,2]
                name_event = event_list[event_index].stem
                frames.append(
                {
                    "name": name_event,
                    "url": str(event_list[event_index].relative_to(*event_list[event_index].parts[:5])),
                    "timestamp": rgb_timestamps[i,0]/1000000
                })
                frameGroups_current.append(name_event)
            #radar
            if sensor_dict["radar"]:    
                radar_index = file_idx[i,3]    
                name_radar = radar_list[radar_index].stem + "_radar"
                frames.append(
                    {
                        "name": name_radar,
                        "url": str(radar_list[radar_index].relative_to(*radar_list[radar_index].parts[:5])),
                        "timestamp": rgb_timestamps[i,0]/1000000
                    }
                )
                frameGroups_current.append(name_radar)
            frameGroups.append(
                {
                    "name": str(i),
                    "videoName": "a",
                    "timestamp": rgb_timestamps[i,0]/1000000,
                    "frames": frameGroups_current
                }
            )
        file_list = {
            "frames": frames,
            "frameGroups": frameGroups
        }

        # Serializing json
        json_object = json.dumps(file_list, indent=4)
    
        # Writing Config
        configuration_file = data_folder / "Main_config.json"
        with open(str(configuration_file), "w") as outfile:
            outfile.write(json_object)  

        
        
        # Write sensor config file

        sensors = []
        i = 0
        for key in sensor_dict:
            if sensor_dict[key]:
                if key == "radar":
                    assert i == len(sensor_dict)-1, "Radar must be last sensor"
                    sensors.append(
                        {
                            "id": i,
                            "name": key,
                            "type": "image",
                            "radar": "BEV"
                        }
                    )
                else:
                    sensors.append(
                        {
                            "id": i,
                            "name": key,
                            "type": "image",
                        }
                    )
                i += 1
        sensor_configuration_file = data_folder / "Sensor_config.json"

        sensors_object = json.dumps(sensors, indent=4)

        with open(str(sensor_configuration_file), "w") as outfile:
            outfile.write(sensors_object)

        print("Test")
        
    