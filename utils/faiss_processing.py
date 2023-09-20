# pip install faiss
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
# pip install langdetect

import numpy as np
import faiss
import glob
import json
import matplotlib.pyplot as plt
import os
import math
from utils.nlp_processing import Translation
# import clip
import torch
import pandas as pd
import re
from langdetect import detect

class File4Faiss:
  def __init__(self, root_database: str):
    self.root_database = root_database

  def re_shot_list(self, shot_list, id, k):
    len_lst = len(shot_list)
    if k>=len_lst or k == 0:
      return shot_list

    shot_list.sort()
    index_a = shot_list.index(id)

    index_get_right = k // 2
    index_get_left = k - index_get_right

    if index_a - index_get_left < 0:
      index_get_left = index_a
      index_get_right = k - index_a
    elif index_a + index_get_right >= len_lst:
      index_get_right = len_lst - index_a - 1
      index_get_left = k - index_get_right

    output = shot_list[index_a - index_get_left: index_a] + shot_list[index_a: index_a + index_get_right]
    return output

  def write_json_file(self, json_path: str, shot_frames_path: str, option='full'):
    count = 0
    self.infos = []
    des_path = os.path.join(json_path, "dict/keyframes_id.json")
    keyframe_paths = sorted(glob.glob(f'{self.root_database}/Keyframes_L*'))
    # print(keyframe_paths)
    # exit()

    for kf in keyframe_paths:
      video_paths = sorted(glob.glob(f"{kf}/*"))
      # print(video_paths)

      for video_path in video_paths:
        image_paths = sorted(glob.glob(f'{video_path}/*.jpg'))

        ###### Get all id keyframes from video_path ######
        id_keyframes = np.array([int(id.split('/')[-1].replace('.jpg', '')) for id in image_paths])
        # print(id_keyframes)
        
        ###### Get scenes from video_path ######
        video_info = video_path.split('/')[-1]
        # print(video_info)
        # exit()
        
        with open(f'{shot_frames_path}/{video_info}.txt', 'r') as f:
          lst_range_shotes = f.readlines()
        lst_range_shotes = np.array([re.sub('\[|\]', '', line).strip().split(' ') for line in lst_range_shotes]) #.astype(np.uint32)

        for im_path in image_paths:
          # im_path = 'Database/' + '/'.join(im_path.split('/')[-3:])
          id = int(im_path.split('/')[-1].replace('.jpg', ''))
          # print(im_path)
          # print(id)
          # exit()
          
          i = 0
          flag=0
          # print(lst_range_shotes)
          # exit()
          for range_shot in lst_range_shotes:
            i+=1
            first, end = range_shot
            first = int(re.sub(',', '', first))
            end = int(end)

            if int(first) <= id <= int(end):
              break
            
            if i == len(lst_range_shotes):
              flag=1
          
          if flag == 1:
            print(f"Skip: {im_path}")
            print(first, end)
            continue

          ##### Get List Shot ID #####
          lst_shot = id_keyframes[np.where((id_keyframes>=first) & (id_keyframes<=end))]
          # print(lst_shot)
          lst_shot = self.re_shot_list(list(lst_shot), id, k=6)
          lst_shot = [f"{i:0>6d}" for i in lst_shot]
          # print(lst_shot)
          # exit()

          ##### Get List Shot Path #####
          lst_shot_path = []
          for id_shot in lst_shot:
            info_shot = {
                "shot_id": id_shot,
                "shot_path": '/'.join(im_path.split('/')[:-1]) + f"/{id_shot}.jpg"
            }
            lst_shot_path.append(info_shot) 

          ##### Merge All Info #####
          info = {
                  "image_path": im_path,
                  "list_shot_id": lst_shot,
                  "list_shot_path": lst_shot_path
                 }
                  
          if option == 'full':        
            self.infos.append(info)   
          else:
            if id == (end+first)//2:
              self.infos.append(info)  

          count += 1
    # exit()
    id2img_fps = dict(enumerate(self.infos))
    
    with open(des_path, 'w') as f:
      f.write(json.dumps(id2img_fps))

    print(f'Saved {des_path}')
    print(f"Number of Index: {count}")

  def load_json_(self, json_path: str):
    with open(json_path, 'r') as f:
      js = json.loads(f.read())

    return {int(k):v for k,v in js.items()}

  def write_bin_file(self, bin_path: str, json_path: str, method='L2', feature_shape=256): 
    count = 0
    # print(json_path)
    id2img_fps = self.load_json_(json_path)

    if method in 'L2':
      index = faiss.IndexFlatL2(feature_shape)
    elif method in 'cosine':
      index = faiss.IndexFlatIP(feature_shape)
    else:
      assert f"{method} not supported"
    
    for _, value in id2img_fps.items():
      image_path = value["image_path"]
      video_name = image_path.split('/')[-2] + '.npy'
      # print(video_name)

      video_id = re.sub('_V\d+', '', image_path.split('/')[-2])
      batch_name = image_path.split('/')[-3].split('_')[-1]
      # clip_name = f"CLIPFeatures_{video_id}_{batch_name}"
      bert_name = './bert_obj_extract_feature'

      feat_path = os.path.join(bert_name, video_name) 
      print(feat_path)
      # exit()

      feats = np.load(feat_path)

      ids = os.listdir(re.sub('/\d+.jpg','',image_path))
      ids = sorted(ids, key=lambda x:int(x.split('.')[0]))

      id = ids.index(image_path.split('/')[-1])
      
      feat = feats[id]
      feat = feat.astype(np.float32).reshape(1,-1)
      # print(feat.shape)
      # exit()
      index.add(feat)
      
      count += 1
    
    # faiss.write_index(index, os.path.join(bin_path, f"faiss_blip_v1_{method}.bin"))
    # exit()
    faiss.write_index(index, os.path.join(bin_path, f"faiss_bert_{method}.bin"))

    print(f'Saved {os.path.join(bin_path, f"faiss_bert_{method}.bin")}')
    print(f"Number of Index: {count}")

def load_json_file(json_path: str):
      js = json.load(open(json_path, 'r'))

      return {int(k):v for k,v in js.items()}
    
def load_bin_file(bin_file: str):
    return faiss.read_index(bin_file)
  
def write_csv(infos_query, des_path):
    check_files = []
    
    ### GET INFOS SUBMIT ###
    for info in infos_query:
      video_name = info['image_path'].split('/')[-2]
      lst_frames = info['list_shot_id']

      for id_frame in lst_frames:
        check_files.append(os.path.join(video_name, id_frame))
    ###########################
    
    check_files = set(check_files)

    if os.path.exists(des_path):
        df_exist = pd.read_csv(des_path, header=None)
        lst_check_exist = df_exist.values.tolist()      
        check_exist = [info[0].replace('.mp4','/') + f"{info[1]:0>6d}" for info in lst_check_exist]

        ##### FILTER EXIST LINES FROM SUBMIT.CSV FILE #####
        check_files = [info for info in check_files if info not in check_exist]
    else:
      check_exist = []

    video_names = [i.split('/')[0] + '.mp4' for i in check_files]
    frame_ids = [i.split('/')[-1] for i in check_files]

    dct = {'video_names': video_names, 'frame_ids': frame_ids}
    df = pd.DataFrame(dct)

    if len(check_files) + len(check_exist) < 99:
      df.to_csv(des_path, mode='a', header=False, index=False)
      print(f"Save submit file to {des_path}")
    else:
      print('Exceed the allowed number of lines')

  
def extract_feats_from_bin(bin_file, idx_image):
    index = faiss.read_index(bin_file)
    feats = []
    ids = []

    for idx in idx_image:
        # print(idx)
        feat = index.reconstruct(idx)
        feats.append(feat)
        ids.append(idx)

    feats = np.vstack(feats)
    return ids, feats

# import numpy as np
   
def save_feats_to_bin(ids, feats, output_bin_path):
    index = faiss.IndexFlatIP(256)
    index.add(feats)
    faiss.write_index(index, output_bin_path)
    arr_idx = ' '.join(str(id) for id in ids)
    # Ghi chuỗi vào tệp tin
    with open('search_continues/list_index.txt', 'w') as file:
        file.write(arr_idx)
    print('done')

def mapping_index(a, b):
    mapped_array = []
    for index in b:
        mapped_array.append(a[index])
    return mapped_array

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Xác định đường dẫn tới thư mục LAVIS
lavis_dir = os.path.join(current_dir, 'LAVIS')

# Thêm đường dẫn tương đối của thư mục LAVIS vào sys.path
sys.path.append(lavis_dir)
from lavis.models import load_model_and_preprocess

# create_file = File4Faiss('Database')
# create_file.write_bin_file(bin_path='./dict/', json_path='./dict/keyframes_id.json', method='cosine', feature_shape=768) # Bert model
# def main():
  
  ### CREATE JSON AND BIN FILES #####
  

  # ##### TESTING #####
  # bin_file='dict/faiss_cosine.bin'
  # json_path = '/dict/keyframes_id.json'

  # cosine_faiss = MyFaiss('./Database', bin_file, json_path)

  # ##### IMAGE SEARCH #####
  # i_scores, _, infos_query, i_image_paths = cosine_faiss.image_search(id_query=9999, k=9)
  # # cosine_faiss.write_csv(infos_query, des_path='/content/submit.csv')
  # cosine_faiss.show_images(i_image_paths)

  # ##### TEXT SEARCH #####
  # text = 'Người nghệ nhân đang tô màu cho chiếc mặt nạ một cách tỉ mỉ. \
  #       Xung quanh ông là rất nhiều những chiếc mặt nạ. \
  #       Người nghệ nhân đi đôi dép tổ ong rất giản dị. \
  #       Sau đó là hình ảnh quay cận những chiếc mặt nạ. \
  #       Loại mặt nạ này được gọi là mặt nạ giấy bồi Trung thu.'

  # scores, _, infos_query, image_paths = cosine_faiss.text_search(text, k=9)
  # # cosine_faiss.write_csv(infos_query, des_path='/content/submit.csv')
  # cosine_faiss.show_images(image_paths)
