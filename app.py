from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from utils.nlp_processing import Translation
import pandas as pd
import json
from pathlib import Path
from posixpath import join
import faiss
from langdetect import detect
from utils.faiss_processing import image_search_faiss, text_search_faiss, write_csv, extract_feats_from_bin, save_feats_to_bin,load_json_file,load_bin_file,mapping_index
from utils.submit import write_csv, show_csv
# from utils.bert_processing import BERTSearch
from utils.ocr_processing import fill_ocr_results, fill_ocr_df
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Xác định đường dẫn tới thư mục LAVIS
lavis_dir = os.path.join(current_dir, 'LAVIS')

# Thêm đường dẫn tương đối của thư mục LAVIS vào sys.path
sys.path.append(lavis_dir)
from lavis.models import load_model_and_preprocess
# http://0.0.0.0:5001/thumbnailimg?index=0

# app = Flask(__name__, template_folder='templates', static_folder='static')
app = Flask(__name__, template_folder='templates')

# Faiss
# bin_file='dict/faiss_blip_v1_cosine.bin'
json_path = 'dict/keyframes_id.json'
json_id2img_path = 'dict/dict_image_path_id2img.json'
json_img2id_path = 'dict/dict_image_path_img2id.json'
json_keyframe2id = 'dict/keyframe_path2id.json'
json_keyframe2path = 'dict/keyframe_id2path.json'
file_path = 'search_continues/list_index.txt'

# with open("dict/info_ocr.txt", "r", encoding="utf8") as fi:
#     ListOcrResults = list(map(lambda x: x.replace("\n",""), fi.readlines()))

# with open("dict/info_asr.txt", "r", encoding="utf8") as fi: 
#     ListASRResults = list(map(lambda x: x.replace("\n",""), fi.readlines()))
# df_asr = pd.read_csv("dict/info_asr.txt", delimiter=",", header=None)
# df_asr.columns = ["video_id", "frame_id", "asr"]    
        
with open(json_id2img_path, 'r') as f:
    DictId2Img = json.loads(f.read())

with open(json_img2id_path, 'r') as f:
    DictImg2Id = json.loads(f.read())

with open(json_keyframe2path, 'r') as f:
    DictKeyframe2Path = json.loads(f.read())

with open(json_keyframe2id, 'r') as f:
    DictKeyframe2Id = json.loads(f.read())

   
LenDictPath = len(load_json_file(json_path))
DictImagePath = load_json_file(json_path)
# BERT
# MyBert = BERTSearch(dict_bert_search='dict/keyframes_id_bert.json', bin_file='dict/faiss_bert.bin', mode='search')

@app.route('/thumbnailimg')
def thumbnailimg():
    print("load_iddoc")

    # global new_bin_file
    # bin_file = new_bin_file if new_bin_file else 'dict/faiss_blip_v1_cosine.bin'
    # CosineFaiss = MyFaiss('Database', bin_file, json_path)
    # DictImagePath = load_json_file(json_path)
    # LenDictPath = len(load_json_file(json_path))
    
    # remove old file submit 
    submit_path = join("submission", "submit.csv")
    old_submit_path = Path(submit_path)
    
    if old_submit_path.is_file():
        os.remove(submit_path)
        # open(submit_path, 'w').close()
    temp_faiss_path = join("search_continues", "temp_faiss.bin")
    old_temp_path = Path(temp_faiss_path)
    if old_temp_path.is_file():
        os.remove(old_temp_path)
    
    temp_txt_path = join("search_continues", "list_index.txt")
    old_temp_path = Path(temp_txt_path)
    if old_temp_path.is_file():
        os.remove(old_temp_path)
        
    # bin_file = 'dict/faiss_blip_v1_cosine.bin'
    print("LenDictPath: ", LenDictPath)
    
    pagefile = []
    index = int(request.args.get('index'))
    if index == None:
        index = 0

    imgperindex = 100
    
    # imgpath = request.args.get('imgpath') + "/"
    pagefile = []

    page_filelist = []
    list_idx = []
    # print(index)
    if LenDictPath-1 > index+imgperindex:
        first_index = index * imgperindex
        last_index = index*imgperindex + imgperindex

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index]["image_path"])
            list_idx.append(tmp_index)
            tmp_index += 1    
    else:
        first_index = index * imgperindex
        last_index = LenDictPath

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index]["image_path"])
            list_idx.append(tmp_index)
            tmp_index += 1    
    # print(page_filelist)
    # print(list_idx)
    # exit()
    for imgpath, id in zip(page_filelist, list_idx):
        pagefile.append({'imgpath': imgpath, 'id': id})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)

@app.route('/imgsearch')
def image_search():
    print("image search")
    pagefile = []
    id_query = int(request.args.get('imgid'))
    k = request.args.get('topk')
    k = int(k[3:])
        
    temp_faiss_path = join("search_continues", "temp_faiss.bin")
    faiss_path = Path(temp_faiss_path)
    if faiss_path.is_file():
        print("continue searchinggg................................................................")
        bin_file = 'search_continues/temp_faiss.bin'
        k= k-200
        
    else:
        bin_file = 'dict/faiss_blip_v1_cosine.bin' 
        
    index = load_bin_file(bin_file)
    query_feats = index.reconstruct(id_query).reshape(1,-1)
    
    scores, idx_image = index.search(query_feats, k=k)
    idx_image = idx_image.flatten()
    scores = scores.flatten()
    
    # Check search continues
    temp_txt_path = join("search_continues", "list_index.txt")
    txt_path = Path(temp_txt_path)
    if txt_path.is_file(): 
        # Đọc dữ liệu từ tệp tin
        with open('search_continues/list_index.txt', 'r') as file:
            data_str = file.read()
        data_list = data_str.split()
        data_array = [int(num) for num in data_list]
        idx_image = mapping_index(data_array, idx_image)
    
    id2img_fps = DictImagePath
    infos_query = list(map(id2img_fps.get, list(idx_image)))
    image_paths = [info['image_path'] for info in infos_query]
    scores = np.array(scores, dtype=np.float32).tolist()
    # print(image_paths)
    
    # _, list_ids, _, list_image_paths = image_search_faiss(index, DictImagePath,id_query, k=200)
    print("searching.......")
    imgperindex = 100 

    for imgpath, id, score in zip(image_paths, idx_image, scores):
        pagefile.append({'imgpath': imgpath, 'id': int(id), 'score':score})
    print("searching.........")
    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)


@app.route('/textsearch')
def text_search():
    print("text search")
    k = str(request.args.get('topk'))
    k = int(k[3:])
    
    temp_faiss_path = join("search_continues", "temp_faiss.bin")
    faiss_path = Path(temp_faiss_path)
    if faiss_path.is_file():
        print("continue searchinggg................................................................")
        bin_file = 'search_continues/temp_faiss.bin'
        k = k-1
        
    else:
        bin_file = 'dict/faiss_blip_v1_cosine.bin'
        
    pagefile = []
    text_query = request.args.get('textquery')
    index = load_bin_file(bin_file)
    
    
    if detect(text_query) == 'vi':
        translater = Translation()
        text = translater(text_query)
    else:
        text = text_query

    __device = "cuda" if torch.cuda.is_available() else "cpu"
    # self.model, preprocess = clip.load("ViT-B/16", device=self.__device)
    model, vis_processors_blip, text_processors_blip = load_model_and_preprocess("blip_image_text_matching", 
                                                                                      "base", 
                                                                                      device=__device, 
                                                                                      is_eval=True)
    
    ###### TEXT FEATURES EXACTING ######
    # text = clip.tokenize([text]).to(__device)  
    # text_features = self.model.encode_text(text).cpu().detach().numpy().astype(np.float32)
    txt = text_processors_blip["eval"](text)
    text_features = model.encode_text(txt, __device).cpu().detach().numpy()

    ###### SEARCHING #####
    scores, idx_image = index.search(text_features, k=k)
    idx_image = idx_image.flatten()
    scores = scores.flatten()
    
    # Check search continues
    temp_txt_path = join("search_continues", "list_index.txt")
    txt_path = Path(temp_txt_path)
    if txt_path.is_file(): 
        # Đọc dữ liệu từ tệp tin
        with open('search_continues/list_index.txt', 'r') as file:
            data_str = file.read()
        data_list = data_str.split()
        data_array = [int(num) for num in data_list]
        idx_image = mapping_index(data_array, idx_image)
        
    # _, idx_image2 = faiss.read_index(bin_file).search(text_features, k=k)
    # print(type(idx_image2))
    # print(type(index))
    
    ###### GET INFOS KEYFRAMES_ID ######
    id2img_fps = DictImagePath
    infos_query = list(map(id2img_fps.get, list(idx_image)))
    image_paths = [info['image_path'] for info in infos_query]
    
    # _, list_ids, _, list_image_paths = text_search_faiss(index, json_path, text_query, k)

    imgperindex = 100 
    scores = np.array(scores, dtype=np.float32).tolist()
    # print(scores)

    for imgpath, id, score in zip(image_paths, idx_image, scores):
        pagefile.append({'imgpath': imgpath, 'id': int(id), 'score':score})

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)

@app.route('/searchcontinues', methods=['POST'])
def search_continues():
    # pagefile = request.files['pagefile']
    data = request.get_json()
    pagefile = data['pagelist']
    # print(pagefile)
    ids = []
    for item in pagefile:
        ids.append(int(item['id']))
    
    new_bin_file = './search_continues/temp_faiss.bin'
    # print(ids)
    # exit()
    bin_file = 'dict/faiss_blip_v1_cosine.bin'
    ids, feats = extract_feats_from_bin(bin_file, ids)
    save_feats_to_bin(ids, feats, new_bin_file)
    print('Saved new bin file')
    imgperindex = 100
    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    return render_template('index_thumb.html', data=data)

@app.route('/neighborsearch')
def neightbor_search():
    print('neightbor frame search')
    pagefile = []
    id_query = int(request.args.get('imgid'))

    
    list_shot_path = DictImagePath[id_query]['list_shot_path']
    
    imgperindex = 100 
    for shot_info in list_shot_path:
        pagefile.append({'imgpath': shot_info['shot_path'], 'id': int(DictKeyframe2Id[shot_info['shot_path']])})

    # show  around 200 key image
    frame_path = DictImagePath[id_query]["image_path"]
    video_dir = frame_path.split("/")[-2]
    keyframe_dir = video_dir.split('_')[0]
    image_name = frame_path.split("/")[-1]


    total_image_in_video = int(DictImg2Id[keyframe_dir][video_dir]["total_image"])
    number_image_id_in_video = int(DictImg2Id[keyframe_dir][video_dir][image_name])

    first_index_in_video = number_image_id_in_video-50 if number_image_id_in_video-50>0 else 0
    last_index_in_video = number_image_id_in_video+50 if number_image_id_in_video+50<total_image_in_video else total_image_in_video
    frame_index = first_index_in_video
    while frame_index < last_index_in_video:
        new_frame_name = DictId2Img[keyframe_dir][video_dir][str(frame_index)]
        frame_in_video_path =  join("Database", "Keyframes_"+keyframe_dir, video_dir, new_frame_name)
        frame_in_video_path =  frame_in_video_path.replace("\\","/")
        if frame_in_video_path in DictKeyframe2Id:
            frame_id_in_video_path = DictKeyframe2Id[frame_in_video_path]
            pagefile.append({'imgpath': frame_in_video_path, 'id': int(frame_id_in_video_path)})

        frame_index += 1

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)

    
@app.route('/asrsearch')
# def asrsearch():
#     print("asr search")
#      # remove old file submit 

#     pagefile = []
#     text_query = request.args.get('text_asr')
#     _, list_ids, _, list_image_paths = MyBert.bert_search(text_query, k=100)

#     imgperindex = 100 

#     for imgpath, id in zip(list_image_paths, list_ids):
#         imgpath = imgpath.replace("\\","/")
#         pagefile.append({'imgpath': imgpath, 'id': int(DictKeyframe2Id[imgpath])})

#     data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
#     return render_template('index_thumb.html', data=data)

# @app.route('/ocrfilter')
# def ocrfilter():
#     print("ocr search")

#     pagefile = []
#     text_query = request.args.get('text_ocr')

#     list_all = fill_ocr_results(text_query, ListOcrResults)
#     list_all.extend(fill_ocr_results(text_query, ListASRResults))

#     # list_all = fill_ocr_df(text_query, df_ocr)
#     # list_all = np.vstack((list_all, fill_ocr_df(text_query, df_ocr)))
    
#     print("list results of ocr + asr: ", list_all)

#     imgperindex = 100 

#     for frame in list_all:
#         list_frame_name = frame.split("/")
#         keyframe_dir = list_frame_name[0][:7]
#         video_dir = list_frame_name[0]
#         new_frame_name = list_frame_name[-1]
#         frame_in_video_path =  join("Database", "KeyFrames"+keyframe_dir, video_dir, new_frame_name)
#         frame_in_video_path =  frame_in_video_path.replace("\\","/")
#         # print("frame_in_video_path: ", frame_in_video_path)
#         if frame_in_video_path in DictKeyframe2Id:
#             print("frame_in_video_path: ", frame_in_video_path)
#             frame_id_in_video_path = DictKeyframe2Id[frame_in_video_path]
#             pagefile.append({'imgpath': frame_in_video_path, 'id': int(frame_id_in_video_path)})

#     data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
#     return render_template('index_thumb.html', data=data)

@app.route('/writecsv')
def submit():
    print("writecsv")
    info_key = request.args.get('info_key')
    mode_write_csv = request.args.get('mode')
    print("info_key", info_key)
    print("mode: ", mode_write_csv)
    info_key = info_key.split(",")

    id_query = int(info_key[0])
    selected_image = info_key[1]
    
    number_line, list_frame_id = write_csv(DictImagePath, mode_write_csv, selected_image, id_query, "submission")
    
    str_fname = ",".join(list_frame_id[:])
    # str_fname += " #### number csv line: {}".format(number_line)

    info = {
        "str_fname": str_fname,
        "number_line": str(number_line)
    }

    return jsonify(info)

@app.route('/get_img')
def get_img():
    # print("get_img")
    fpath = request.args.get('fpath')
    # fpath = fpath
    list_image_name = fpath.split("/")
    image_name = "/".join(list_image_name[-2:])

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jph")
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.resize(img, (1280,720))

    # print(img.shape)
    # Tọa độ và kích thước hình chữ nhật nền
    x, y = 30, 80
    w, h = cv2.getTextSize(image_name, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)[0]

    # Vẽ hình chữ nhật nền
    cv2.rectangle(img, (x, y), (x+w, y+h), (192, 192, 192), -1)

    img = cv2.putText(img, image_name, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)
    # img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
    #                3, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dowload_submit_file', methods=['GET'])
def dowload_submit_file():
    print("dowload_submit_file")
    filename = request.args.get('filename')
    fpath = join("submission", filename)
    print("fpath", fpath)

    return send_file(fpath, as_attachment=True)

@app.route('/get_first_row')
def getFirstRowOfCsv():
    csv_path = "submission/submit.csv"
    result = {
        'video_id':"None",
        'frame_id':"None"
    }
    if os.path.exists(csv_path):
        lst_frame = show_csv(csv_path)[0]
        video_id, frame_id = lst_frame.split("/")[-2:]
        result["video_id"] = video_id
        result["frame_id"] = int(frame_id[:-4])

    return result

@app.route('/visualize')
def visualize():
    number_of_query = int(request.args.get('number_of_query'))
    csv_path = join("submission", "query-{}.csv".format(number_of_query))

    query_path = join("query","query-{}.txt".format(number_of_query))
    if os.path.exists(query_path):
        with open(query_path, "rb") as fi:
            query_content = fi.read().decode("utf-8").replace(" ","_")

    pagefile = []
    lst_frame = show_csv(csv_path)
    for frame_path in lst_frame:
        frame_id = DictKeyframe2Id[frame_path]
        pagefile.append({'imgpath': frame_path, 'id': int(frame_id)})
    if query_content is not None:
        data = {'num_page': 1, 'pagefile': pagefile, 'query': query_content}
    else:
        data = {'num_page': 1, 'pagefile': pagefile}

    return render_template('index_thumb.html', data=data)

@app.route('/search_image_path')
def search_image_path():
    pagefile = []
    frame_path = request.args.get('frame_path')
    list_frame_split = frame_path.split("/")
    
    video_dir = list_frame_split[0]
    image_name = list_frame_split[1] + ".jpg"
    keyframe_dir = video_dir.split('_')[0]
    

    frame_path = join("Database", "Keyframes_"+keyframe_dir, video_dir, image_name)
    frame_path = frame_path.replace("\\","/")
    frame_id = DictKeyframe2Id[frame_path]
    
    imgperindex = 100 
    pagefile.append({'imgpath': frame_path, 'id': int(frame_id)})

    # show  around 40 key image
    total_image_in_video = int(DictImg2Id[keyframe_dir][video_dir]["total_image"])
    number_image_id_in_video = int(DictImg2Id[keyframe_dir][video_dir][image_name])

    first_index_in_video = number_image_id_in_video-50 if number_image_id_in_video-50>0 else 0
    last_index_in_video = number_image_id_in_video+50 if number_image_id_in_video+50<total_image_in_video else total_image_in_video
    frame_index = first_index_in_video
    while frame_index < last_index_in_video:
        new_frame_name = DictId2Img[keyframe_dir][video_dir][str(frame_index)]
        frame_in_video_path =  join("Database", "Keyframes_"+keyframe_dir, video_dir, new_frame_name)
        frame_in_video_path =  frame_in_video_path.replace("\\","/")
        if frame_in_video_path in DictKeyframe2Id:
            frame_id_in_video_path = DictKeyframe2Id[frame_in_video_path]
            pagefile.append({'imgpath': frame_in_video_path, 'id': int(frame_id_in_video_path)})

        frame_index += 1

    data = {'num_page': int(LenDictPath/imgperindex)+1, 'pagefile': pagefile}
    
    return render_template('index_thumb.html', data=data)


if __name__ == '__main__':
    submit_dir = "submission"
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)

    app.run(debug=False, host="0.0.0.0", port=5001)
