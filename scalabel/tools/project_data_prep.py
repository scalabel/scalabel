import argparse
from pathlib import Path
import csv
import numpy as np
import json
import os
from project_pointcloud import projectPoints
from event_preparer import EventPreparer


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

# Expected folder structure:
# data_folder
#   - folder_name
#       - rgb
#       - events.h5
#   - folder_name_pointcloud
#       - lidar
#       - lidar_timestamps.txt
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepeare data for Annotation')
    parser.add_argument('data_folder', type=str, help='Path data folder')
    parser.add_argument('folder_name', type=str, help='name of the folder, eg. "rain_day_bias_1"') # TODO: fix this, more elegant solution
    args = parser.parse_args()

    event_time_delta_us = 15000 #TODO: make parser arg

    data_folder = Path(args.data_folder)
    assert data_folder.is_dir(), "Directory {} does not exist".format(str(data_folder)) 
    
    folder_name = args.folder_name
    folder_path = data_folder / folder_name
    assert folder_path.is_dir(), "Directory {} does not exist".format(str(folder_path))

    h5_file_path = folder_path / "events.h5"
    assert h5_file_path.is_file(),  "File {} does not exist".format(str(h5_file_path))

    rgb_folder_path = folder_path / "rgb"
    assert rgb_folder_path.is_dir(), "Directory {} does not exist".format(str(rgb_folder_path))

    folder_path_pointcloud = data_folder / (folder_name + "_pointcloud")
    assert folder_path_pointcloud.is_dir(), "Directory {} does not exist".format(str(folder_path_pointcloud))

    lidar_folder_path = folder_path_pointcloud / "lidar"
    assert lidar_folder_path.is_dir(), "Directory {} does not exist".format(str(lidar_folder_path))

    rgb_timestamps_path = folder_path / "rgb_timestamps.txt"
    assert rgb_timestamps_path.is_file(), "File {} does not exist".format(str(rgb_timestamps_path))

    lidar_timestamps_path = folder_path_pointcloud / "lidar_timestamps.txt"
    assert lidar_timestamps_path.is_file(), "File {} does not exist".format(str(lidar_timestamps_path))
    
    rgb_list = list_full_paths(rgb_folder_path)
    lidar_list = list_full_paths(lidar_folder_path)

    rgb_timestamps = timestamps_reader(rgb_timestamps_path)
    lidar_timestamps = timestamps_reader(lidar_timestamps_path)

    # dont judge, super inefficent, feel free to improve
    #TODO: cases where there are no lidar ect.
    file_idx = np.zeros((len(rgb_timestamps),3),dtype=np.int64)
    for i in range(len(rgb_timestamps[:,0])):
        for j in range(len(lidar_timestamps[:,0])):
            if (rgb_timestamps[i,0] <= lidar_timestamps[j,0]):
                file_idx[i,0] = i
                file_idx[i,1] = j
                break
    
    # make lidar representations
    lidar_projections_path = folder_path_pointcloud / "lidar_img"
    if not os.path.isdir(lidar_projections_path):
        os.makedirs(lidar_projections_path)

    # if (len(list_full_paths(lidar_projections_path)) < len(lidar_list)):
    #     projectPoints(lidar_list[90:120], lidar_projections_path)

    # make event representations
    event_representations_path = folder_path / "event_img"
    if not os.path.isdir(event_representations_path):
        os.makedirs(event_representations_path)
    
    
    event_prep = EventPreparer(h5_file_path, event_time_delta_us)
    first_event = event_prep.get_start_time_us()
    last_event = event_prep.get_final_time_us()
    if (sum(rgb_timestamps[:,0]>=first_event)>0):
        first_index = np.argmax(rgb_timestamps[:,0]>=first_event)
    else:
        print("Error") #TODO: better error
    if (True): #TODO: FIX
        last_index = np.argmin(rgb_timestamps[:,0]<=last_event)
    else:
        print("Error")
    #event_prep.create_representations(rgb_timestamps[first_index:last_index,0], event_representations_path)
    file_idx[first_index:last_index,2] = np.array(range(last_index-first_index))

    event_list = list_full_paths(event_representations_path)
    lidar_img_list = list_full_paths(lidar_projections_path)
    # make json
    # TODO: make it not dependent on first event
    frames = []
    frameGroups = []
    for i in file_idx[first_index:last_index,0]:
        lidar_index = file_idx[i,1]
        event_index = file_idx[i,2]
        if(lidar_index>115):
            break
        name_rgb = rgb_list[i].stem
        name_lidar = lidar_img_list[lidar_index].stem
        name_event = event_list[event_index].stem
        
        frames.append(
            {
                "name": name_rgb,
                "url": str(rgb_list[i].relative_to(*rgb_list[i].parts[:5])),
                "timestamp": rgb_timestamps[i,0]/1000000
            })
        frames.append(
            {
                "name": name_lidar,
                "url": str(lidar_img_list[lidar_index].relative_to(*lidar_img_list[lidar_index].parts[:5])),
                "timestamp": rgb_timestamps[i,0]/1000000
            })
        frames.append(
            {
                "name": name_event,
                "url": str(event_list[event_index].relative_to(*event_list[event_index].parts[:5])),
                "timestamp": rgb_timestamps[i,0]/1000000
            })
        
        frameGroups.append(
            {
                "name": str(i),
                "videoName": "a",
                "timestamp": rgb_timestamps[i,0]/1000000,
                "frames": [
                    name_rgb,
                    name_lidar,
                    name_event
                ]
            }
        )
    file_list = {
        "frames": frames,
        "frameGroups": frameGroups
    }

    # Serializing json
    json_object = json.dumps(file_list, indent=4)
 
    # Writing to sample.json
    configuration_file = data_folder / (folder_name + "_config.json")
    with open(str(configuration_file), "w") as outfile:
        outfile.write(json_object)  

    print("Test")
    