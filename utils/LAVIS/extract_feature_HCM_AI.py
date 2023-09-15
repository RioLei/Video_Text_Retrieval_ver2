
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import argparse

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

import numpy as np
import os


def extract_helper(model, vis_processors_blip, data_path, feature_dest_path):

    images = []
    list_dir = os.listdir(data_path)
    list_dir.sort()
    # print(f"list_dir:{list_dir}")
    # exit()
    for filename in [filename for filename in list_dir if filename.endswith(".jpg")]:
        print(filename)
        img_path = os.path.join(data_path, filename)
        # print(img_path)
        raw_image = Image.open(img_path).convert("RGB")
        img = vis_processors_blip["eval"](raw_image).unsqueeze(0).to(device)

        image_features = model.encode_image(img).detach().cpu().numpy()

        images.append(image_features)
    

    print(np.asarray(images).shape)


    print(f"feature_dest_path:{feature_dest_path}")
    np.save(feature_dest_path, np.asarray(images))
    print("Save done!\n")
    

def extract(model, vis_processors_blip, 
            keyframe_folder, feature_dest_path):

  list_dir = os.listdir(keyframe_folder)
  list_dir.sort()
  # print(list_dir)
  for dir in list_dir:
    print(dir)
    data_path = os.path.join(keyframe_folder, dir)
    dest_path = os.path.join(feature_dest_path, dir+".npy")
    extract_helper(model, vis_processors_blip, data_path, dest_path)

if __name__=="__main__":

  # Tạo một đối tượng ArgumentParser
    parser = argparse.ArgumentParser(description='Process a video and specify the destination folder.')

    # Thêm đối số keyframe_folder
    parser.add_argument('--keyframe_folder', type=str, default="/mnt/HDD/Chau_Truong/AIC 2023/data/file zip/Videos/Videos_L01/video", 
                        help='Path to the keyframe file')

    # Thêm đối số feature_dest_path
    parser.add_argument('--feature_dest_path', type=str, default='', 
                        help='Path to the destination folder')
    
    # # Phân tích đối số từ dòng lệnh
    args = parser.parse_args()

    # keyframe_folder = "/mnt/HDD/Chau_Truong/AIC 2023/Video-Text-Retrieval/Database/Keyframes_L01"
    # feature_dest_path = "/mnt/HDD/Chau_Truong/AIC 2023/BLIP/AIC_BLIP_features"
    print("start training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_blip, vis_processors_blip, text_processors_blip = load_model_and_preprocess("blip_image_text_matching", 
                                                                                      "base", 
                                                                                      device=device, 
                                                                                      is_eval=True)

    extract(model_blip, vis_processors_blip, args.keyframe_folder, args.feature_dest_path)

    




