import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import argparse

# def save_to_csv(data_array, output_file):
#     # Create an empty DataFrame
#     df = pd.DataFrame()

#     # Iterate through each dictionary in the data array
#     for item in data_array:
#         video_id = item['video_id']
#         tags = item['tags']

#         # Create a dictionary for each row
#         row_data = {'video_id': video_id}
#         for tag in tags:
#             row_data[tag] = int(1)

#         # Append the row to the DataFrame
#         df = df.append(row_data, ignore_index=True)

#     # Fill missing values with 0
#     df = df.fillna(int(0))
    
#     int_columns = df.columns.drop('video_id')
#     df[int_columns] = df[int_columns].astype(int)

#     # Save the DataFrame to a CSV file
#     df.to_csv(output_file, index=False)


def get_unique_detection_class_entities(root_directory):
    re = []
    
    # Loop through each file in the folder
    for folder in os.listdir(root_directory):
        file_retrival = {}
        file_retrival[folder] = []
        # print(folder)
        folder_path = root_directory+'/'+ folder
        # exit()
        for file in os.listdir(folder_path):
            count_dict = {}
            file_path = folder_path+'/'+ file
            # print(file_path)
            unique_entities = set()
            
            if file_path.endswith('.json'):
                with open(file_path, 'r') as json_file:
                    try:
                        data = json.load(json_file)
                        name = file_path.split('/')[-1].split('.')[0] + '.jpg'  # L01_V001/000000
                        # print(name)
                        # exit()

                        # Get the detection_class_entities from the JSON data
                        entities = data.get('detection_class_entities', [])
                            
                        # Add the entities to the set
                        unique_entities.update(entities)
                        # print(type(unique_entities))
                        # Count the occurrences of each entity
                        for entity in entities:
                            count_dict[entity] = count_dict.get(entity, 0) + 1
                        # print(count_dict)
                        temp = []
                        for item in unique_entities:
                            temp.append(f"{count_dict[item]} {item}")
                        # print(temp)
                        # exit()
                        file_data = {'keyframe_id': name, 'tags': ', '.join(list(temp))}
                        
                        file_retrival[folder].append(file_data)
                        # file_retrival[folder]['tags'] 
        
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON file: {file_path}")
        re.append(file_retrival)

    return re

def extract_features_bert(model, root_dir, feature_dest_path):
    i = 0
    
    re = get_unique_detection_class_entities(root_dir)
    for _, dirs, _ in os.walk(root_dir):
        if dirs != []:
            break
    # print(dirs) # list folders, Ex: ['L01_V001', 'L02_V001',....]
    for item in re: # list item for each folder 
        list_data = item[dirs[i]] # contain list keyframe and tags. Ex: [{000000.jpg, tags}, {},...] 
        txt_emb = []
        # For each folder
        for data in list_data: # {000000.jpg, tags} 
                        
            keyframe_id = data['keyframe_id'] # "000000.jpg"
            # print(dirs[i] + '/' + keyframe_id)
            tags = data['tags'] # tags of the keyframe id "000000.jpg"
            # print(tags)
            text_features = model.encode(tags)
            txt_emb.append(text_features) # keyframe 
            txt_emb.append(text_features) # frame lan can
            txt_emb.append(text_features) # frame lan can
        # exit()
        print("feature_dest_path: " + feature_dest_path + '/' + dirs[i] + '.npy')
        np.save(feature_dest_path + '/' + dirs[i] + '.npy', txt_emb)
        print("Save done!\n")
        i += 1

    return txt_emb

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process a video and specify the destination folder.')

    parser.add_argument('--object_folder', type=str, default="./objects", 
                        help='Path to the object file')

    parser.add_argument('--feature_dest_path', type=str, default='./bert_obj_extract_feature', 
                        help='Path to the destination folder')

    args = parser.parse_args()
    
    print("start training")
    
    folder_path = './objects'
    feature_dest_path = './bert_obj_extract_feature'
    model = SentenceTransformer('all-distilroberta-v1')

    txt_emb = extract_features_bert(model, folder_path, feature_dest_path)
    
