   # -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:51:42 2022

@author: sbhat
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import sys
import random
from tqdm import tqdm
from google.protobuf.json_format import MessageToDict

class CustomLandmarkList():
    def __init__(self,landmark=[]):
        self.landmark=landmark
        
class MPUtils():
    def __init__(self, save_video=False, play_video=False,body_pose = True, high_res_hands = True, out_dir=os.getcwd()):
        """
        Parameters
        ----------
        save_video : Bool, optional
            Flag to save hand tracked video. The default is False.
        play_video : Bool, optional
            Flag to play hand tracked video. The default is False.
        body_pose : TYPE, optional
            DESCRIPTION. The default is True.
        high_res_hands : TYPE, optional
            DESCRIPTION. The default is True.
        out_dir : str path, optional
            DESCRIPTION. The default is os.getcwd().

        Returns
        -------
        None.

        """
        self.save_video = save_video
        self.play_video = play_video
        self.out_dir = out_dir
        self.body_pose = body_pose
        self.high_res_hands = high_res_hands
        
        
    def rotate_image(self,image, width, height, angle = 45):
        """
        Parameters
        ----------
        img : TYPE
            DESCRIPTION.
        width : TYPE
            DESCRIPTION.
        height : TYPE
            DESCRIPTION.

        Returns
        -------
        rotated_image : TYPE
            DESCRIPTION.

        """
        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width/2, height/2)
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        # rotate the image using cv2.warpAffine
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        
        return rotated_image
    
    def translate_image(self,image, width, height, T):
        """
        Parameters
        ----------
        image : TYPE
            DESCRIPTION.
        width : TYPE
            DESCRIPTION.
        height : TYPE
            DESCRIPTION.

        Returns
        -------
        img_translation : TYPE
            DESCRIPTION.

        """
        
        # We use warpAffine to transform
        # the image using the matrix, T
        img_translation = cv2.warpAffine(image, T, (width, height))
        
        return img_translation
    
    def draw_landmarks(self,landmarks,image,bbox=False):
        """
        Parameters
        ----------
        landmarks : NormalizedLandmarkList object(Mediapipe object)
            Landmarsk to draw.
        image : ndarray (image array)
            Image without hand bounding boxes and landmarks.

        Returns
        -------
        image : ndarray
            Image with bounding boxes around hands and landmarks.

        """
        height, width, _ = image.shape
        for i in range(len(landmarks.landmark)):
            pt1 = landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
    
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        
        if bbox:
            h, w, c = image.shape
            cx_min=  w
            cy_min = h
            cx_max= cy_max= 0
            for id, lm in enumerate(landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if cx<cx_min:
                    cx_min=cx
                if cy<cy_min:
                    cy_min=cy
                if cx>cx_max:
                    cx_max=cx
                if cy>cy_max:
                    cy_max=cy
            cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)   
        return image
    
    def adjust_gamma(self,image, gamma=1.5):
        """
        Parameters
        ----------
        image : ndarray
            Image input.
        gamma : float, optional
            Factor by which to increase gamma. The default is 1.5.

        Returns
        -------
        ndarray
            Image with adjusted gamma.

        """
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    
    def get_augmentation_params(self, img_size):
        
        rotate = random.getrandbits(1)
        translate =random.getrandbits(1)
        w , h = img_size
        
        angle = random.uniform(0,180)
        t_height, t_width = random.uniform(-h/20,h/20) , random.uniform(-w/20,w/20)  
        T = np.float32([[1, 0, t_width], [0, 1, t_height]]) 
        
        return (rotate, translate) , (angle, T)
    
    def getKeyPoints(self,video, augment=False):
        """
        Parameters
        ----------
        video : str path
            Path to video file.
        
        Returns
        -------
        keypoints : dict
            Dictionary of lists for {'handedness':[],'keypoints':[],'frame_number':[]}.
        fps : int
            Frames Per Second.
        length : int
            number of frames in video.
        
        Note
        ----
        Only works with video, designed to be used to create training dataset.
        """
        
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(video)
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
        mp_holistic = mp.solutions.holistic
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) 
        size = (frame_width, frame_height)
        
        if augment:
            print("Augmenting.")
            aug_params = self.get_augmentation_params(size)
            (rotate, translate),(angle, T) = aug_params 
            print(f"Augmentation Params:{aug_params[0]}") 
        else:
            rotate , translate = False, False
            angle = 0 
            T = []
            aug_params = (rotate,translate),(angle,T)
            
        if self.save_video :
            tracked_dir = os.path.join(self.out_dir,'tracked_videos')
            os.makedirs(tracked_dir,exist_ok=True) 
            tracked_fname = os.path.basename(video).split('.')[0] + '_tracked.avi'
            tracked_save_path = os.path.join(tracked_dir,tracked_fname)
            result = cv2.VideoWriter(tracked_save_path, 
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 fps, size)
        
        #   Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            sys.exit()
        
        ## set up pose API
        # pose = mp_pose.Pose(min_detection_confidence=0.5,
        #                     min_tracking_confidence=0.5)
        # hands = mp_hands.Hands(min_detection_confidence=0.5,
        #                        min_tracking_confidence=0.5)
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, 
                                        min_tracking_confidence=0.5)
        
        ## TODO: add case for no high_res_hands
        ## ENUMS to seperate TORSO + LEFT/RIGHT ARM pose keypoints
        RIGHT_ARM = [11,12,14,16,23,24] 
        LEFT_ARM = [11,12,13,15,23,24]
        ## ENUMS to remove ladnmarks below torso
        LEGS = [i for i in range(26,33)]
        
        if not self.high_res_hands:
            RIGHT_ARM.append([18,20,22])
            LEFT_ARM.append([17,19,21])
            
        ## Read until video is completed
        ## Count Frame number
        frame_number = 0
        
        ## Save handedness and keypoints in a dict
        keypoints = {'handedness':[],'keypoints':[],'frame_number':[]}
    
        length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=length,position=0,desc =f"Processing Video: {os.path.basename(video)}" ,leave=True)
        while(cap.isOpened()):  
            # Capture frame-by-frame
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            # image = cv2.flip(image,1)
            height, width, _ = image.shape
            image = self.adjust_gamma(image)
            if augment :
                if rotate :
                    image = self.rotate_image(image=image, width=width, height=height, angle=angle)
                if translate:
                    image = self.translate_image(image, width, height, T)
                
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = holistic.process(image)
            ## Process body pose
            right = []
            left = []
            
            if self.high_res_hands:
                if results.right_hand_landmarks:
                    for hand_landmarks in results.right_hand_landmarks.landmark:
                        right.append(hand_landmarks)
                if results.left_hand_landmarks:
                    for  hand_landmarks in results.left_hand_landmarks.landmark:
                        left.append(hand_landmarks)
                        
            if self.body_pose and self.high_res_hands:
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
            
            if self.body_pose and not self.high_res_hands:
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
                ## Add landmarks to dict
                keypoints['handedness'].append('Left')
                keypoints['keypoints'].append(left_landmarks)
                keypoints['frame_number'].append(frame_number)
            if len(right) > 0 :
                right_landmarks = CustomLandmarkList(right)
                keypoints['handedness'].append('Right')
                keypoints['keypoints'].append(right_landmarks)
                keypoints['frame_number'].append(frame_number)
            
            if self.save_video or self.play_video:
                # image = pose_utils.draw_text(image, text)
                if len(right) > 0 :
                    image = self.draw_landmarks(right_landmarks,image)
                if len(left) > 0 :
                    image = self.draw_landmarks(left_landmarks,image)
                        
            pbar.update(1)
            ## PLAY VIDEO WITH KEYPOINTS
            if self.play_video:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = cv2.flip(image,1)
                cv2.imshow('MediaPipe Pose and Hands',image)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            ## SAVE VIDEO WITH KEYPOINTS
            if self.save_video:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result.write(image)
            frame_number += 1
        pbar.close()
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        return keypoints , fps, length, aug_params