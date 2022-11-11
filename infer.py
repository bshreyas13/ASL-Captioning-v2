# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 23:00:46 2022

@author: sbhat
"""
import sys
import os
import argparse
import numpy as np
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
from tensorflow.keras.models import load_model
##CUSTOM MODULES
from utils.DatasetLab import DatasetUtils
from utils.ClassifierUtils import Utils
from utils.MediapipeUtils import MPUtils
from utils.MediapipeUtils import CustomLandmarkList

## Method to parse args from commnad line 
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", help="path to inference config.", default=f"{os.path.join(os.getcwd(),'configs', 'LSTM_config_infer.yaml')}")
    args.add_argument("--out_dir", "-o", help="output directory ", default=f"{os.path.join(os.path.dirname(os.getcwd()),'WORKDIR','saved_inference')}")  
    args = args.parse_args()
    return args.config, args.out_dir

if __name__=="__main__":
    ## Read input
    config_path, out_dir = parse_args()
    os.makedirs(out_dir,exist_ok=True) 
    
    ## Setup tools
    config_tool = Utils(out_dir)
    pose_utils = MPUtils(out_dir = out_dir)
    data_tool = DatasetUtils(None, out_dir)
        
    ## LOAD: Configs
    config_list = config_tool.load_model_configs(config_path)
    model_name,mode,saved_model,wsize,stride,comment,in_mode,save_video,play_video = config_list
    
    body_pose = True
    high_res_hands = True
    
    ## Check and select stream
    if in_mode == 'webcam':
        in_stream = 0
    else:
        in_stream = in_mode
    
    ## LOAD: Saved model
    gesture_clf = load_model(saved_model)
    gesture_clf.summary()
    
    ## Load label info to convert to name label
    label_info = data_tool.read_annotations_df(os.path.join(os.path.dirname(out_dir),'label_info.csv'))
    _,label_names_dict = config_tool.get_binary_dict(label_info)
    
    
    ## ENUMS to seperate TORSO + LEFT/RIGHT ARM pose keypoints
    RIGHT_ARM = [11,12,14,16,23,24]
    LEFT_ARM = [11,12,13,15,23,24]
    ## ENUMS to remove ladnmarks below torso
    LEGS = [i for i in range(26,33)]
    if not high_res_hands:
        RIGHT_ARM.append([18,20,22])
        LEFT_ARM.append([17,19,21]) 
    
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(in_stream)
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    size = (frame_width, frame_height)
    
    wsize = int(wsize*fps)
    stride = int(stride*fps) + 1
    
    text = ''
    
    if save_video :
        assert in_stream != 0
        tracked_dir = os.path.join(out_dir)
        os.makedirs(tracked_dir,exist_ok=True) 
        tracked_fname = os.path.basename(in_stream).split('.')[0] + '_predicted.avi'
        tracked_save_path = os.path.join(tracked_dir,tracked_fname)
        result = cv2.VideoWriter(tracked_save_path, 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             fps, size)
    ## Get keypoints based on window_size and stride 
    ## Predict gesture (write on screen)
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, 
                                    min_tracking_confidence=0.5)
    
    #   Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        sys.exit()
    ## Read until video is completed
    ## Count Frame number
    frame_number = 0
    ## Save handedness and keypoints in a dict
    keypoints_L = []
    keypoints_R = []

    while(cap.isOpened()):
        # Capture frame-by-frame
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            if in_stream == 0 :
                continue
            else:
                break
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.flip(image,1)
        height, width, _ = image.shape
                
        image = pose_utils.adjust_gamma(image)
        
        ## Separate ladmarks to left and right 
        results = holistic.process(image)
        ## Process body pose
        right = []
        left = []
        
        if high_res_hands:
            if results.right_hand_landmarks:
                for hand_landmarks in results.right_hand_landmarks.landmark:
                    right.append(hand_landmarks)
            if results.left_hand_landmarks:
                for  hand_landmarks in results.left_hand_landmarks.landmark:
                    left.append(hand_landmarks)
                    
        if body_pose and high_res_hands:
            if results.pose_landmarks:
                ## Separate ladmarks to left and right 
                for idx, landmark in enumerate (results.pose_landmarks.landmark):
                    if idx in LEGS:
                        continue
                    if idx in RIGHT_ARM:
                        if results.right_hand_landmarks:
                            right.append(landmark)
                    if idx in LEFT_ARM:
                        if results.left_hand_landmarks:
                            left.append(landmark)
        
        if body_pose and not high_res_hands:
            if results.pose_landmarks:
                ## Separate ladmarks to left and right 
                for idx, landmark in enumerate (results.pose_landmarks.landmark):
                    if idx in LEGS:
                        continue
                    if idx in RIGHT_ARM:
                        right.append(landmark)
                    if idx in LEFT_ARM:
                        left.append(landmark)
        if len(left) > 0 :               
            left_landmarks = CustomLandmarkList(left)

        if len(right) > 0 :
            right_landmarks = CustomLandmarkList(right)
        
        keypoints_L.append(data_tool.landmark_to_point_vector(left_landmarks))
        if len(keypoints_L) == wsize:
            l_vector=np.expand_dims(np.array(keypoints_L),axis=0)
            pred_left = gesture_clf.predict(l_vector)
            label_left = label_names_dict[str(np.where(pred_left >= np.max(pred_left), 1, 0).flatten())]
            keypoints_L = keypoints_L[stride:]
            
            if save_video or play_video:
                ## LEFT TEXT
                font = cv2.FONT_HERSHEY_SIMPLEX
                # fontScale
                fontScale = 1
                color = (255, 0, 0)
                # Line thickness of 2 px
                thickness = 2                        
                # org
                text_l = "Left " + label_left
                org = (int(0.8 * frame_width), int(0.3 * frame_height))
                # Using cv2.putText() method
                image = cv2.putText(image, text_l, org, font, 
                                   fontScale, color, thickness, cv2.LINE_AA)
        
        # keypoints_R.append(data_tool.landmark_to_point_vector(right_landmarks))
        # if len(keypoints_R) == wsize:
        #     r_vector=np.expand_dims(np.array(keypoints_R),axis=0)
        #     pred_right = gesture_clf.predict(r_vector)
        #     label_right = label_names_dict[str(np.where(pred_right >= np.max(pred_right), 1, 0).flatten())]
        #     keypoints_R = keypoints_R[stride:]
            
        #     if save_video or play_video:
        #         ## RIGHT TEXT
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         # fontScale
        #         fontScale = 1
        #         color = (255, 0, 0)
        #         # Line thickness of 2 px
        #         thickness = 2
        #         text_r = "Right " + label_right
        #         org = (int(0.2 * frame_width), int(0.3 * frame_height))
        #         # Using cv2.putText() method
        #         image = cv2.putText(image, text_r, org, font, 
        #                            fontScale, color, thickness, cv2.LINE_AA)
        if save_video or play_video:
            # image = pose_utils.draw_text(image, text)
            if len(left) > 0 : 
                image = pose_utils.draw_landmarks(right_landmarks,image)
            if len(left) > 0 : 
                image = pose_utils.draw_landmarks(left_landmarks,image)
        if play_video:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.flip(image,1)
            cv2.imshow('MediaPipe Pose and Hands',image)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        ## SAVE VIDEO WITH KEYPOINTS
        if save_video:
            # image = cv2.flip(image,1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result.write(image)
        frame_number += 1
    cap.release()
    if save_video:
        result.release()
    # Closes all the frames
    cv2.destroyAllWindows()