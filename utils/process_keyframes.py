import os
import cv2 
import glob
import pandas as pd
from tqdm import tqdm

def resize_keyframes(Database_path):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    img_paths = glob.glob(f'{Database_path}/Keyframes*/*/*.jpg')
    # print(img_paths)
    # exit()

    for img_path in img_paths:
        print("img_path: ", img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (384,384))

        os.system(f'rm {img_path}')
        cv2.imwrite(img_path, img)
        
def reformat_keyframe_name(list_csv_paths:str, list_obj_path:str):
    """
    It takes a list of csv files and a list of frame paths, and renames the frames in the frame paths to
    match the csv files
    
    :param list_csv_paths: the path to the folder containing the csv files. If folder contains Batch1 and Batch2 csv then function will rename all frame in Batch1 and Batch2.
    :param list_obj_path: the path to the folder containing the frames
    """
    lst_csv = glob.glob(f'{list_csv_paths}/*.csv')
    # print("lst_csv: ", lst_csv)
    lst_csv.sort()
    dct_names = {}
    # exit()

    for csv_path in tqdm(lst_csv):
        df = pd.read_csv(csv_path)
        for i in df.index:
            row = df.iloc[i]
            # print(row)
            video_id = csv_path.split('/')[-1][:-4]
            # print(video_id)

            if video_id.split('_')[0] == 'L18':
                key = f'{video_id}/{int(row[0]):03}.json' # --> only L18
                # print(key)
            else:
                key = f'{video_id}/{int(row[0]):04}.json'  #  --> not include L18
                # print(key)
            # print(key)
            value = f'{video_id}/{int(row[3]):06}.json' 
            # print(value)
            dct_names[key] = value
            # print(dct_names)
    #         exit()

    # exit()

    for key, value in tqdm(dct_names.items()):
        # print(key)
        # keyframe = f'KeyFrames_{key.split("/")[0][:3]}' # KeyFramesC00_V00
        # print(keyframe)
        frame_src_path = f'{list_obj_path}/{key}'
        print(frame_src_path)
        frame_dst_path = f'{list_obj_path}/{value}'
        print(frame_dst_path)
        # exit()

        if frame_src_path == frame_dst_path or not os.path.exists(frame_src_path):
            continue
            
        lst_frame_in_video = os.listdir('/'.join(frame_src_path.split('/')[:-1]))
        # print(lst_frame_in_video)
        # exit()

        # if prev_keyframe != keyframe:
        #     lst_frame_in_video = os.listdir('/'.join(frame_src_path.split('/')[:-1]))
        #     prev_keyframe = keyframe
        # exit()
        if frame_dst_path.split('/')[-1] in lst_frame_in_video:
            os.remove(frame_src_path)
        else:
            os.rename(frame_src_path, frame_dst_path)
            print('Done!')

# list_csv_path = './map-keyframes'
# list_obj_path = './objects'
# reformat_keyframe_name(list_csv_path, list_obj_path)

import os

def process_folders(path):
    # Lặp qua các thư mục trong đường dẫn
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        
        # Kiểm tra nếu là thư mục
        if os.path.isdir(folder_path):
            for folder in os.listdir(folder_path):
                fol = os.path.join(folder_path, folder)
                
                for f in os.listdir(fol):
                    # print(f)
                    # exit()
                    if f.endswith('_keyframe.jpg'):
                        print("Processing folder:", fol)
                        old_file_path = os.path.join(fol, f)
                        # print(old_file_path)
                        # exit()
                        new_file_name = f.replace('_keyframe', '')
                        new_file_path = os.path.join(fol, new_file_name)
                        os.rename(old_file_path, new_file_path)
                        print("Renamed file:", old_file_path, "to", new_file_path)

from utils.faiss_processing import load_json_file,load_bin_file

json_path = 'dict/keyframes_id.json'
DictImagePath = load_json_file(json_path)

bin_file = 'dict/faiss_blip_v1_cosine.bin' 

def remove_noise_images(root_data, bin_file, id_remove, k=2000):

    index = load_bin_file(bin_file)
    query_feats = index.reconstruct(id_remove).reshape(1,-1)
    
    scores, idx_image = index.search(query_feats, k=k)
    idx_image = idx_image.flatten()
    scores = scores.flatten()
    
    
    
    id2img_fps = DictImagePath
    infos_query = list(map(id2img_fps.get, list(idx_image)))
    image_paths = [info['image_path'] for info in infos_query]
    print(image_paths)



# Gọi hàm để xử lý các thư mục trong đường dẫn "Database/"
# path = "./Database"
# process_folders(path)
# resize_keyframes(path)

# id_remove = 1

# remove_noise_images(path, bin_file, id_remove)
