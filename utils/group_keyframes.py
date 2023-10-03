import re
def extractKeyframes(input_string):
    pattern = r'/([^/]+)/([^/]+)/\d+\.jpg$'
    match = re.search(pattern, input_string)
    return match.group(2)

# def rearrangeDict(dict):
#     grouped_data = {}
#     for name in dict:
#         keyframe_dir = extractKeyframes(name['imgpath'])
#         list_frame = name
#         if keyframe_dir not in grouped_data:
#             grouped_data[keyframe_dir] = {"keyframe_dir": keyframe_dir, "list_frame": []}
#         grouped_data[keyframe_dir]["list_frame"].append(name)
#     output_dict = list(grouped_data.values())
#     return output_dict

def convertArray(array):
    converted_array = []
    for item in array:
        imgpath = item['imgpath']
        video_id = imgpath.split('/')[-2]
        new_item = {'video_id': video_id, 'list_frame': [item]}
        existing_item = next((x for x in converted_array if x['video_id'] == video_id), None)
        if existing_item:
            existing_item['list_frame'].append(item)
        else:
            converted_array.append(new_item)
    return converted_array

def deleteFrames(ids, dict, text_out):
    for i in ids:
        for j in dict:
            if j['keyframe_dir'] == i:
                dict.remove(j)
    data_str = ', '.join(map(str, dict))
    with open(text_out, 'w') as file:
        file.write(dict)
    return()

# ids = ['L01_V001', 'L02_V001',...]
# dict = input_dict
# text_out = output txt file
import os
import json
import csv
def extractMetadata(filepath, outputfile):
    outputlist = []
    listpath = os.listdir(filepath)
    for name in listpath:
        namepath = filepath+ "\\" +name
        videoid = name.split('.')[0]
        with open(namepath, "r", encoding = "utf8") as file:
            data = json.load(file)
        value = [data['watch_url'], videoid]
        outputlist.append(value)
    with open(outputfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(outputlist)
    return("save done")

import os
import pandas
from pandas import read_csv
import csv

def extractMapKeyframes(filepath, outputfile):
    listpath = os.listdir(filepath)
    for name in listpath:
        path = filepath+ "\\"+name
        df = read_csv(path)
        df.drop(['n', 'fps'], axis =1, inplace = True )
        Keyframes = name.split("_")[0]
        Value = name.split(".")[0]
        df['frame_idx']= df['frame_idx'].apply(lambda x: f"Database/Keyframes_{Keyframes}/{Value}/"+ "{:06}".format(x) +".jpg")
        df= df[['frame_idx', 'pts_time']]
        df.to_csv(outputfile, mode = 'a', header= False, index = False)
    return('save done')

