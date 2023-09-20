import os
import cv2
import subprocess
import math
import argparse


def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

def save_specific_frame(cap, frame_no, folder, is_keyframe=0):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    print("frame_no:", frame_no)
    # print("type of frame_no:", type(frame_no))
    
    if is_keyframe == 0:
      if len(str(frame_no))<6:
        outname = os.path.join(folder,"0"*(6-len(str(frame_no)))+str(frame_no)+'.jpg')
      else:
        outname = os.path.join(folder,str(frame_no)+'.jpg')
    else:
      if len(str(frame_no))<6:
        # print("0"*(6-len(str(frame_no)))+str(frame_no))
        # print("str(frame_no)=",str(frame_no))
        # print("folder:", folder)
        outname = os.path.join(folder,"0"*(6-len(str(frame_no)))+str(frame_no)+'_keyframe.jpg')

      else:
        outname = os.path.join(folder,str(frame_no)+'_keyframe.jpg')
    if(os.path.exists(outname)):
      print("Path exist:", outname)
      return
    cv2.imwrite(outname, frame)
    print ('Saved: '+outname)

def save_i_keyframes(video_fn, dest_folder):
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0]
        cap = cv2.VideoCapture(video_fn)
        for i in range(len(i_frames)-1):
            value = i_frames[i:i+2]
            folder = dest_folder
            keyframe = value[0]
            frame_25_percent = value[0] + math.floor((value[1]-value[0])*33/100)
            # print("frame 25 percent:",frame_25_percent)
            # frame_50_percent = value[0] + math.floor((value[1]-value[0])*50/100)
            # print("frame 50 percent:",frame_50_percent)
            frame_75_percent = value[0] + math.floor((value[1]-value[0])*66/100)

            save_specific_frame(cap, keyframe, folder, is_keyframe=1)
            save_specific_frame(cap, frame_25_percent, folder)
            # save_specific_frame(cap, frame_50_percent, folder)
            save_specific_frame(cap, frame_75_percent, folder)
            
        cap.release()
    else:
        print ('No I-frames in '+video_fn)

if __name__ =="__main__":
    # Tạo một đối tượng ArgumentParser
    parser = argparse.ArgumentParser(description='Process a video and specify the destination folder.')

    # Thêm đối số filename
    parser.add_argument('--video_path', type=str, default="/mnt/HDD/Chau_Truong/AIC 2023/data/file zip/Videos/Videos_L01/video", 
                        help='Path to the video file')

    # Thêm đối số dest_folder
    parser.add_argument('--keyframes_root', type=str, default='', 
                        help='Path to the destination folder')
    
    # # Phân tích đối số từ dòng lệnh
    args = parser.parse_args()

    # video_path = "/mnt/HDD/Chau_Truong/AIC 2023/data/file zip/Videos/Videos_L01/video"
    video_root_name = args.video_path.split("/")[-2].split("_")[-1] #   video_root_name: L01

    # keyframes_root = "/mnt/HDD/Chau_Truong/AIC 2023/data/file zip/Database_with_more_frames"

    keyframes_path = os.path.join(args.keyframes_root, "Keyframes_"+video_root_name)

    if os.path.isdir(keyframes_path)==False:
      os.mkdir(keyframes_path)

    video_dir = os.listdir(args.video_path)
    
    video_dir.sort()
    # print(video_dir)
    # exit()
    video_names = [video_name for video_name in video_dir if video_name.endswith(".mp4")]
    # print(f"video_names:{video_names}")
    # exit()
    for video_name in video_names:
      # print(video_name)
      individual_video_path  = os.path.join(args.video_path, video_name)
      # print(f"individual_video_path:{individual_video_path}")
      video_name_without_extension = os.path.splitext(os.path.basename(individual_video_path))[0]
      # print(f"video_name_without_extension:{video_name_without_extension}")

      keyframe_dest_folder = os.path.join(keyframes_path, video_name_without_extension)
      # print(f"keyframe_dest_folder:{keyframe_dest_folder}")
      if os.path.isdir(keyframe_dest_folder)==False:
        os.mkdir(keyframe_dest_folder)

      save_i_keyframes(individual_video_path, keyframe_dest_folder)
