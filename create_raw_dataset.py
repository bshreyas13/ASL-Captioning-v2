# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:41:40 2022

@author: sbhat
"""
import os
import pandas as pd
import argparse
import sys
import numpy as np
import random
from sklearn.preprocessing import LabelBinarizer
## CUSTOM MODULES
from utils.MediapipeUtils import MPUtils
from utils.ClassifierUtils import Utils
from utils.DatasetLab import DatasetUtils
from utils.utils import get_clips_by_label

## Method to parse args from commnad line 
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", help="path to configs directory.", default=f"{os.path.join(os.getcwd(),'configs','LSTM_config_infer.yaml')}")
    args.add_argument("--data_dir", "-d", help="data directory ", default=f"{os.path.join(os.path.dirname(os.getcwd()),'Data')}") 
    args.add_argument("--out_dir", "-o", help="output directory ", default=f"{os.path.join(os.path.dirname(os.getcwd()),'WORKDIR')}") 
    args.add_argument("--annotation_filename", "-a_name", help="output filename ", default="Annotations.csv") 
    args.add_argument("--extract_clips", "-ec", help="Flag to extract and save clips by label for each video ", default= False)
    
    args = args.parse_args()
    return args.data_dir, args.out_dir, args.annotation_filename, args.extract_clips, args.config
    
if __name__=="__main__":
    
    in_dir , out_dir, out_filename, extract_clips, config = parse_args()
    os.makedirs(out_dir,exist_ok=True)
    
    config_tool = Utils(out_dir, raw_dataset=True)
    all_configs = config_tool.load_model_configs(config)   
    body_pose, high_res_hands, classes = all_configs
  
    ## Init UTILS
    tool = DatasetUtils(in_dir, out_dir)
    pose_tool = MPUtils(out_dir = out_dir, save_video=True)
    
    ## CHECK workdir(AKA out_dir)
    workdir = os.listdir(out_dir)
    if 'RAW_Dataset' in workdir:
        dataset_dir = os.path.join(out_dir,'RAW_Dataset')
        ## Check file already processed
        _extracted = os.listdir(dataset_dir)
        _extracted = [item.split('.')[0] for item in _extracted]
    else:
        dataset_dir = None
    
    
    ## Load Annotation if exists
    ## Else init a annotation file with the list of videos in Data Directory
    try:
        ann_dir = os.path.join(out_dir,'Annotations')
        ann_files = os.listdir(ann_dir)
    except Exception as ex:
        print(f"Error while reading Annotation file: {ex}")
        tool.init_annotation_file(out_filename,classes)
        ann_files = os.listdir(os.path.join(out_dir,'Annotations'))
        print(f"Initializing an Annotation file.Annotate timestamps for each class in {os.path.join(out_dir,ann_files[0])}")
        print("If you already have a annotation file replace with the annotated Annotations.csv/xlsx file.")
        sys.exit()
    ## Read and parse Annotation
    annotation_df = tool.read_annotations_df(os.path.join(ann_dir,ann_files[0]))
    
    
    ## Encode categorical labels and save a Look up table
    class_ids = [idx for idx in range(len(classes))]
    label_df = pd.DataFrame(list(zip(class_ids, classes)),
                      columns=['Ids', 'Action'])
    labels = LabelBinarizer().fit_transform(label_df.Action)
    label_info_df = pd.DataFrame(list(zip(class_ids, classes,labels)),
                      columns=['Ids', 'Action','binary_label'])
    
    label_info_df.to_csv(os.path.join(out_dir,'label_info.csv'))
    
    ## Process Videos 
    videos_list = annotation_df["Video_Name"]
    v_list = []
    fps_list = []
    num_frames_list = []
    path_list = [] 
    aug_params_list = []
    print(f'Total Number of Videos in {ann_files[0]}: {len(videos_list)}')
    for idx, video in enumerate(videos_list):    
        video_path = os.path.join(in_dir,video)
        annotation = annotation_df[annotation_df['Video_Name']== video]
        
        ### Get timecodes for each class in video
        valid, timecodes, handedness_codes = tool.parse_annotation(annotation,video,classes, exclusions=['Video_Name','Note'])
        
        ### check for errors in annotation
        if not valid:
            print(f"Ignoring {video}, invalid label")
        
        elif dataset_dir is not None and video.split('.')[0] in _extracted:
            print(f"{video} already processed. Vector exists. Skipping!")
            continue
        
        else:
            
            keypoints,fps, num_frames, aug_params = pose_tool.getKeyPoints(video = video_path, augment=False)                        
            ## Update video info
            v_list.append(video)
            path_list.append(video_path)
            fps_list.append(fps)
            num_frames_list.append(num_frames)
            aug_params_list.append(aug_params)

            ## Make vectors with features as X and Encoded Labels as Y
            # each frame in X has [x,y,z,frame_number]
            Y=[]
            X=[]
            for idx , frame_number in enumerate(keypoints['frame_number']):
                ## Interpret timecodes to frames
                foi = {}
                for entry in classes:
                    frames = tool.timecode_to_frames(timecodes[entry], fps)
                    foi[entry] = frames
                    if frame_number in frames:                
                        X.append(tool.landmark_to_point_vector(keypoints['keypoints'][idx],frame_number))     
                        Y.append(int(label_df[label_df.Action==entry].Ids))

            ## TODO: FIX this , puts stop and other gestures into None causing bad predictions
            ## Update annotation
                    
            if extract_clips:
                get_clips_by_label(video_path,foi,out_dir)
                
            X_ = np.asarray(X)
            Y_ = np.asarray(Y)
            DATA_VECTOR = np.column_stack((X_,Y_))
            save_name = video.split('.')[0]+'.csv'
            tool.save_vector(DATA_VECTOR,dataset_dir, save_name)
    
            ## save video info includes frame rate and num frames and augmentation info
            video_info_df = pd.DataFrame(list(zip(v_list,fps_list,num_frames_list,path_list,aug_params_list)),
                                          columns=['video_name','fps','num_frames','video_path','augmentation_info'])
            video_info_df.to_csv(os.path.join(out_dir,'video_info.csv'))
            
            