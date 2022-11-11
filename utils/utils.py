# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 21:11:59 2022

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
from tqdm import tqdm

def run_inference(in_stream,wsize,stride,save_video,out_dir,pose_utils,body_pose):
    pass
    """
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(in_stream)
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    
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
        tracked_fname = os.path.basename(in_stream).split('.')[0] + '_predicted.mp4'
        tracked_save_path = os.path.join(tracked_dir,tracked_fname)
        result = cv2.VideoWriter(tracked_save_path, 
                             cv2.VideoWriter_fourcc(*'mpv4'),
                             fps, size)
    ## Get keypoints based on window_size and stride 
    ## Predict gesture (write on screen)
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5,
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
        
        if body_pose:
            results_pose = pose.process(image)
            ## Process body pose
            if results_pose.pose_landmarks:                    
                ## Separate ladmarks to left and right 
                right = []
                left = []
                for idx, landmark in enumerate (results_pose.pose_world_landmarks.landmark):
                    if idx in LEGS:
                        continue
                    if idx not in RIGHT_ARM:
                        left.append(landmark)
                    if idx not in LEFT_ARM:
                        right.append(landmark)
                left_landmarks = CustomLandmarkList(left)
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
                
                keypoints_R.append(data_tool.landmark_to_point_vector(right_landmarks))
                if len(keypoints_R) == wsize:
                    r_vector=np.expand_dims(np.array(keypoints_R),axis=0)
                    pred_right = gesture_clf.predict(r_vector)
                    label_right = label_names_dict[str(np.where(pred_right >= np.max(pred_right), 1, 0).flatten())]
                    keypoints_R = keypoints_R[stride:]
                    
                    if save_video or play_video:
                        ## RIGHT TEXT
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # fontScale
                        fontScale = 1
                        color = (255, 0, 0)
                        # Line thickness of 2 px
                        thickness = 2
                        text_r = "Right " + label_right
                        org = (int(0.2 * frame_width), int(0.3 * frame_height))
                        # Using cv2.putText() method
                        image = cv2.putText(image, text_r, org, font, 
                                           fontScale, color, thickness, cv2.LINE_AA)
                        
                # if save_video or play_video:
                #     # image = pose_utils.draw_text(image, text)
                #     image = pose_utils.draw_landmarks(right_landmarks,image)
                #     image = pose_utils.draw_landmarks(left_landmarks,image)
                    
        if high_res_hands:
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    ## Identify and tag landmarks by handedness
                    handedness_dict = MessageToDict(results.multi_handedness[idx])
                    hand_info = handedness_dict['classification'][0]
    
                    if hand_info['label'] == 'Left':
                        keypoints_L.append(data_tool.landmark_to_point_vector(hand_landmarks))
                        if len(keypoints_L) == wsize:
                            in_vector=np.expand_dims(np.array(keypoints_L),axis=0)
                            pred_left = gesture_clf.predict(in_vector)
                            label_left = label_names_dict[str(np.where(pred_left >= np.max(pred_left), 1, 0).flatten())]
                            keypoints_L = keypoints_L[stride:]
                            
                            if save_video or play_video:
                                # image = cv2.flip(image,1)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                # fontScale
                                fontScale = 1
                                color = (255, 0, 0)
                                # Line thickness of 2 px
                                thickness = 2                        
                                # org
                                text = hand_info['label'] + label_left
                                org = (int(0.8 * frame_width), int(0.3 * frame_height))
                                # Using cv2.putText() method
                                image = cv2.putText(image, text, org, font, 
                                                   fontScale, color, thickness, cv2.LINE_AA)
                                # image = cv2.flip(image,1)
                             
                    if hand_info['label'] == 'Right':
                        #print(hand_landmarks)
                        keypoints_R.append(data_tool.landmark_to_point_vector(hand_landmarks))
                        if len(keypoints_R) == wsize:
                            in_vector=np.expand_dims(np.array(keypoints_R),axis=0)
                            pred_right = gesture_clf.predict(in_vector)
                            label_right = label_names_dict[str(np.where(pred_right >= np.max(pred_right), 1, 0).flatten())]
                            keypoints_R = keypoints_R[stride:]
                            
                            if save_video or play_video:
                                # image = cv2.flip(image,1)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                # fontScale
                                fontScale = 1
                                color = (255, 0, 0)
                                # Line thickness of 2 px
                                thickness = 2                        
                                # org
                                text = hand_info['label'] + label_right
                                org = (int(0.2 * frame_width), int(0.3 * frame_height))
                                # Using cv2.putText() method
                                image = cv2.putText(image, text, org, font, 
                                                   fontScale, color, thickness, cv2.LINE_AA)
                                # image = cv2.flip(image,1)
                                
                    if save_video or play_video:
                        # image = pose_utils.draw_text(image, text)
                        image = pose_utils.draw_landmarks(hand_landmarks,image)
        
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

    """
def get_clips_by_label(video,FOI, out_dir):

    # Create a VideoCapture object and read from input file
    # ONLY FOR VIDEO
    cap = cv2.VideoCapture(video)
    
    ## details for video writer
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    size = (frame_width, frame_height)

    save_dir = os.path.join(out_dir,'Data_Clips',os.path.basename(video))
    os.makedirs(save_dir,exist_ok=True)    
    video_writers = {}    
    for label in list(FOI.keys()):
        _fname = os.path.basename(video).split('.')[0]+f'_{label}.avi'
        save_path = os.path.join(save_dir,_fname)
        result = cv2.VideoWriter(save_path, 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             fps, size)
        video_writers[label] = result
        
    #   Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        sys.exit()
    ## Read until video is completed
    ## Count Frame number
    frame_number = 0
    
    length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length,position=0,desc =f"Extracting clips from : {os.path.basename(video)}" ,leave=True)
    while(cap.isOpened()):
        # Capture frame-by-frame
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        for label, frames in FOI.items():
            if frame_number in frames:
                video_writers[label].write(image)
        
        frame_number += 1
        pbar.update(1)
    cap.release()
    for _ , writer in video_writers.items():
        writer.release()
    