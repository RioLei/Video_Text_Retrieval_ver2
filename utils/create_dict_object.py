# from glob import glob
# import os 
# import json 

# def create_dict_image_path(data_dir):
#     dict_json_path = {}
#     for keyframe_dir in sorted(os.listdir(data_dir)):
#         keyframe_dir_path = os.path.join(data_dir, keyframe_dir)
#         dict_json_path[keyframe_dir[-7:]] = {}

#         for subdir in sorted(os.listdir(keyframe_dir_path)):
#             if subdir not in dict_json_path:
#                 dict_json_path[keyframe_dir[-7:]][subdir] = {}

#             subdir_path = os.path.join(keyframe_dir_path, subdir)
#             # print(subdir_path)
#             list_image_path = glob(os.path.join(subdir_path, "*.jpg"))
#             list_image_path.sort()
#             # print(list_image_path)

#             for index, image_path in enumerate(list_image_path):
#                 image_name = image_path.split("'\'")[-1]
#                 image_name = image_name.replace("\\","/")
#                 print(image_name)
#                 dict_json_path[keyframe_dir[-7:]][subdir][int(index)] = image_name # id2img
#                 # dict_json_path[keyframe_dir[-7:]][subdir][image_name] = index # img2id 
            
#             dict_json_path[keyframe_dir[-7:]][subdir]["total_image"] = len(list_image_path)

# path = 'Database/'
# create_dict_image_path(path)
#     # with open('dict_image_path_id2img.json', 'w') as f:
#     #     json.dump(dict_json_path, f)

#     # with open('dict_image_path_img2id.json', 'w') as f:
#     #     json.dump(dict_json_path, f)

import os
import json

def get_unique_detection_class_entities(folder_path):
    re = []
    # Loop through each file in the folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            unique_entities = set()
            file_retrival = {}
            file_path = os.path.join(root, file_name)
            dir_name = root.split('/')[-1]
            # print(dir_name)
            
            # Check if the file is a JSON file
            if file_name.endswith('.json'):
                # print(file_name)
                # exit()
                # Load the JSON data
                with open(file_path, 'r') as json_file:
                    try:
                        data = json.load(json_file)
                        name = dir_name+'/'+file_name.split('.')[0]

                        # Get the detection_class_entities from the JSON data
                        entities = data.get('detection_class_entities', [])
                        
                        # Add the entities to the set
                        unique_entities.update(entities)
                        # print(type(unique_entities))
                        
                        file_retrival['video_path'] = name
                        file_retrival['tags'] = list(unique_entities)
                        re.append(file_retrival)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON file: {file_path}")

    return re

import pandas as pd

def save_to_csv(data_array, output_file):
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate through each dictionary in the data array
    for item in data_array:
        video_id = item['video_id']
        tags = item['tags']

        # Create a dictionary for each row
        row_data = {'video_id': video_id}
        for tag in tags:
            row_data[tag] = int(1)

        # Append the row to the DataFrame
        df = df.append(row_data, ignore_index=True)

    # Fill missing values with 0
    df = df.fillna(int(0))
    
    int_columns = df.columns.drop('video_id')
    df[int_columns] = df[int_columns].astype(int)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    
    
    
folder_path = './Dataset_2023/objects/L02_V018'
output_file = './Dataset_2023/object.csv'
re = get_unique_detection_class_entities(folder_path)

save_to_csv(re, output_file)