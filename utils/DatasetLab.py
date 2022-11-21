# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:25:28 2022

@author: sbhat
"""
import os
import pandas as pd
import sys
import numpy as np
import glob
import re
import random
from tqdm import tqdm
from contextlib import redirect_stdout
import traceback

class DatasetUtils():
    def __init__(self, data_dir, output_dir, comment='_'):
        """
        Parameters
        ----------
        data_dir : str path
            Path to directory of keypoint vectors as csv (For full Videos).
        output_dir : str path
            Path to directory to save all package outputs.

        Returns
        -------
        None.

        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.comment = comment
        self.dataset_dir = os.path.join(self.output_dir, 'Dataset')
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.LOG_PATH = os.path.join(self.output_dir, f'RUN_LOG_{comment}.txt')

    def init_annotation_file(self, annotation_filename,classes):
        """
        Parameters
        ----------
        annotation_filename : str 
            Filename to save annotation file.

        Returns
        -------
        None.

        """
        annotation_df = pd.DataFrame()
        videos_list = [os.path.basename(video)
                       for video in os.listdir(self.data_dir)]
        annotation_df["Video_Name"] = videos_list
        for entry in classes:
            annotation_df[entry]=""
        annotation_df["Note"] = ""
        
        # Save DF as csv
        save_dir = os.path.join(self.output_dir, 'Annotations')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, annotation_filename)
        annotation_df.to_csv(save_path, index=False)

    def read_annotations_df(self, annotation_file_path):
        """


        Parameters
        ----------
        annotation_file_path : str, path
            Path to annotation csv/xlsx file.

        Returns
        -------
        annotation_df : DataFrame
            Pandas DataFrame of Annotations.

        """
        ext = os.path.basename(annotation_file_path).split('.')[-1]
        if ext == 'xlsx':
            annotation_df = pd.read_excel(annotation_file_path)
        elif ext == 'csv':
            annotation_df = pd.read_csv(annotation_file_path)
        else:
            print("Error in annotation file : File must be csv or xlsx")
            sys.exit()

        return annotation_df

    def parse_annotation(self, annotation, video,classes, exclusions=['Video_Name','Note']):
        """
        Parameters
        ---------
        annotation : pandas series (row of annotation_df)
            Annotation for specific video.
        video : str
            Video Name.
        classes: list
            List of classes in the problem setup
        exclusions: list
            List of column names to ignore while finding labels in annotation file
            
        Returns
        -------
        valid : Bool
            TRUE if annotation has errors, FALSE otherwise.
        timecodes : dict
            Dictrionary for lists of timecodes in each class
            example:{'Stop': ['00:01-00:05'], 'Go': ['00:06-00:09', ' 00:09-00:13 '], 'Change_Lane': ['00:13-00:19', ' 00:19-00:24'], 'None': []}
        h_codes : dict 
             Dictionary of handedness codes for each Timecode.
             example: {S:R,RL;G:R;CL:L,L}

        """
        try:
            valid = True
            labels = []
            for entry in annotation.columns:
                if entry in exclusions:
                    continue
                else:
                    labels.append(entry)
            valid = self.check_annotated_classes(labels,classes)
           
            if not valid :
                print("ANNOTATION ERROR:Classes in annotation are not same as classes setup in config file")
                print(f"Annotated Classes:{labels}")
                print(f"Classes setup: {classes}")
                sys.exit()
                
            ## TODO : use re to serach for the word Invalid
            if list(annotation["Note"])[0].split(':')[0] == 'Invalid':
                valid = False
                timecodes = {}
                h_codes = {}

            else:
                timecodes = {}
                checks_ = {}
                for entry in classes:
                    _timecodes = list(annotation[entry].dropna())
                    if len(_timecodes) != 0:
                        _timecodes = str(_timecodes[0]).split(',')
                    timecodes[entry] = _timecodes               
                    # Initialize True for checks for each label 
                    # Used to check for missing handedness codes in annotation
                    checks_[entry] = True
               
                handedness_codes = list(annotation['Note'])
                handedness_codes = handedness_codes[0].split(';')
                h_codes = {}
                for item in handedness_codes:
                    item = item.split(':')
                    b = [b.split(',') for b in item[1:]]
                    if b[0][0] != '':
                        h_codes[item[0]] = b[0]
                    else:
                        h_codes[item[0]] = []
                
                for label, res in checks_.items():
                    code = self.get_char_code(label)
               
                    if len(timecodes[label]) != 0 :
                        if code in h_codes.keys():                         
                            checks_[label] = self.verify_handedness_integrity(
                                timecodes[label], h_codes[code])
                        else:
                            checks_[label] = False
            
                if False in list(checks_.values()):
                    print(f'ANNOTATION ERROR:number of clips != number of handedness_codes. Check entries for {video}.')
                    print('INFO:')
                    print(h_codes.items())
                    print(checks_.items())
                    valid = False
                
        except Exception as ex:
            valid = False
            print(f"Caught exception in DatasetUtils.parse_annotations:{ex}")
            print(traceback.format_exc())
            sys.exit()
            
        return valid, timecodes, h_codes
    
    def get_char_code(self,label):
        name = label.split('_')
        if len(name)>1:
            code =''
            for entry in name:
                char = [*entry][0]
                code += char
        else:
            code = [*name[0]][0]
        return code
    
    def timecode_to_frames(self, timecodes, fps):
        """
        Parameters
        ----------
        timecodes : str
            String time codes from annotation file.
        fps : int
            frames per second.

        Returns
        -------
        dict 
            dict of frames corresponding to timecodes

        """
        frames = {}
        for idx, timecode in enumerate(timecodes):
            if timecode == 'nan' or timecode == ' ':
                continue
            markers = str(timecode).split('-')
            start_secs = str(markers[0]).split(':')
            start_time = int(start_secs[0])*60 + int(start_secs[1])
            stop_secs = str(markers[1]).split(':')
            stop_time = int(stop_secs[0])*60 + int(stop_secs[1])

            frames[idx] = [num for num in range(int(start_time*fps), int(stop_time*fps))]

        return frames

    def verify_handedness_integrity(self, code, h_code_list):
        """
        Parameters
        ----------
        code : TYPE
            DESCRIPTION.
        h_code_list : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return len(code) == len(h_code_list)
                
    def check_annotated_classes(self,labels,classes):
        return set(classes) == set(labels)
    
    def prepare_directories(self, classes):
        """
        Parameters
        ----------
        classes : list
            List of classes in dataset.

        Returns
        -------
        list
            List of paths to each class dirs

        """
        # Get proper save path for each class
        save_paths = []
        for entry in classes:
            class_dir = os.path.join(self.dataset_dir, entry)
            os.makedirs(class_dir, exist_ok=True)
            save_paths.append(class_dir)

        return save_paths

    def landmark_to_point_vector(self, keypoints, frame_number=None):
        """
        Parameters
        ----------
        keypoints : NormalizedLandmark object (Google MP)
            KEYPOINTS FOR A GIVEN FRAME
        frame_number : int
            FRAME NUMBER FOR GIVEN LANDMARKS.

        Returns
        -------
        vector : list, 
            A flattened list of (x,y,z) for 21 kepoints for each hand(63 entries).
            Along with frame number appended at vector[-1] ==> len(vector)=64

        """
        point_list = []
        for pt in keypoints.landmark:
            point_list.append([pt.x, pt.y, pt.z])
        # print(np.array(point_list).shape)
        vector = list(np.asarray(point_list).flatten())
        if frame_number != None:
            vector.append(frame_number)
        return vector

    def save_vector(self, vector, save_dir=os.getcwd(), filename='test.csv'):
        """
        Parameters
        ----------
        vector : ndarray
            Array to save.
        save_dir : str path, optional
            output_path. The default is current working directory.
        filename : str, optional
            Name to save file with. The default is 'test.csv'.

        Returns
        -------
        None.

        """
        # Setup Dataset Directories
        if save_dir == None:
            dataset_dir = os.path.join(self.output_dir, 'RAW_Dataset')
            os.makedirs(dataset_dir, exist_ok=True)
        else:
            dataset_dir = save_dir
        save_path = os.path.join(dataset_dir, filename)
        np.savetxt(save_path, vector, delimiter=',')

    def load_vector(self, filepath):
        """
        Parameters
        ----------
        filepath : str path
            Path to .csv vector.

        Returns
        -------
        vector : ndarray
            ndarray vector.

        """
        vector = np.loadtxt(filepath, delimiter=',')
        return vector

    def split_vector_by_hand(self, vector):
        """
        Parameters
        ----------
        vector : ndarray
            input vector.

        Returns
        -------
        one_hand_vector : ndarray
            DESCRIPTION.
        other_hand_vector : ndarray
            DESCRIPTION.

        """

        frame_numbers = vector[:, -2]
        alt_hand_list = []
        one_idx_list = []
        other_idx_list = []

        for idx, num in enumerate(frame_numbers):
            if num in alt_hand_list:
                other_idx_list.append(idx)
            else:
                alt_hand_list.append(num)
                one_idx_list.append(idx)
        one_hand_vector = []
        other_hand_vector = []
        
        for ind, idx in enumerate(one_idx_list):
            one_hand_vector.append(list(vector[idx, :]))

        for ind, idx in enumerate(other_idx_list):
            other_hand_vector.append(list(vector[idx, :]))
        one_hand_vector = np.asarray(one_hand_vector)
        other_hand_vector = np.asarray(other_hand_vector)

        return one_hand_vector, other_hand_vector

    def get_vectors_by_label(self, vector):
        """
        Parameters
        ----------
        vector : TYPE
            DESCRIPTION.

        Returns
        -------
        vectors_by_label : TYPE
            DESCRIPTION.

        """
        left_V, right_V = self.split_vector_by_hand(vector)
        if left_V.shape[0] == 0 :
            og_vector = right_V
        elif right_V.shape[0] == 0 :
            og_vector = left_V
        else: 
            og_vector = np.concatenate((left_V, right_V))

        df = pd.DataFrame(og_vector)
        a = df.groupby(df.columns[-1])
        dfs = [a.get_group(x) for x in a.groups]
        vectors_by_label = {}
        for idx in range(len(dfs)):
            vectors_by_label[idx] = dfs[idx]

        return vectors_by_label

    def resample_reshape_data(self, win_size, stride, fps, data,
                              save_dir, save_name, LSTM=True, INCLUDE_FRAME_NO=False):
        """
        Parameters
        ----------
        win_size : float
            Window in seconds.
        stride : float
            Stride in seconds.
        fps : int
            Video frequency.
        data : ndarray
            Data vector to resample.
        save_dir : str path
            Output directory
        save_name : str
            Filename.
        INCLUDE_FRAME_NO : Bool, optional
            Flag to include/exclude frame from from X. The default is True.

        Returns
        -------
        None.

        """
        wsize_frames = int(win_size*fps)
        stride_frames = int(fps*stride)

        if INCLUDE_FRAME_NO:
            X = data[:, :-1]
        else:
            X = data[:, :-2]
        Y = data[:, -1]

        if win_size != 0 or stride != 0:
            # if not LSTM no timesteps, all flattened
            if not LSTM:
                count = 0
                for idx in range(0, X.shape[0], stride_frames):
                    X_sample = X[idx:idx+wsize_frames]
                    Y_sample = Y[idx:idx+wsize_frames]
                    # ignore samples less than wsize frames
                    if X_sample.shape[0] == wsize_frames:
                        reshaped_X = X_sample.flatten()
                        reshaped_Y = int(Y_sample.flatten()[0])
                        reshaped_data = np.append(reshaped_X, reshaped_Y)
                        # print(reshaped_data.shape)
                        save_name_ = save_name.split(
                            '.')[0]+f"_{str(count)}"+".csv"
                        self.save_vector(reshaped_data, save_dir, save_name_)
                        count += 1
            else:
                count = 0
                for idx in range(0, X.shape[0], stride_frames):
                    X_sample = X[idx:idx+wsize_frames]
                    Y_sample = Y[idx:idx+wsize_frames]
                    # ignore samples less than wsize frames
                    if X_sample.shape[0] == wsize_frames:
                        Y_sample = np.expand_dims(Y_sample, axis=1)
                        reshaped_data = np.hstack((X_sample, Y_sample))
                        # print(reshaped_data.shape)
                        save_name_ = save_name.split(
                            '.')[0]+f"_{str(count)}"+".csv"
                        self.save_vector(reshaped_data, save_dir, save_name_)
                        count += 1
        else:
            count = 0
            for idx in range(X.shape[0]):
                X_sample = X[idx]
                Y_sample = Y[idx]
                data = np.append(X_sample, Y_sample)
                save_name_ = save_name.split('.')[0]+f"_{str(count)}"+".csv"
                self.save_vector(data, save_dir, save_name_)
                count += 1

    def get_path_list(self, shuffle=True, seed=42):
        """

        Parameters
        ----------
        shuffle : Bool, optional
            Shuffle or not. The default is True.
        seed : int, optional
            Random.random seed. The default is 42.

        Returns
        -------
        data_path_list : list
            List of path corresponding to data.

        """
        path = os.path.join(self.dataset_dir, '**', '*.csv')
        data_path_list = []
        for filepath in glob.iglob(path, recursive=True):
            filename = os.path.basename(filepath)
            check = len(filename.split('.')[0].split('_'))
            if check != 2:
                continue
            data_path_list.append(filepath)
        if shuffle:
            random.Random(seed).shuffle(data_path_list)
        return data_path_list

    def get_df(self, data_paths, classes, label_info):
        """
        Parameters
        ----------
        shuffled_data : TYPE
            DESCRIPTION.
        classes : TYPE
            DESCRIPTION.
        label_info : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """
        data = []
        for path in tqdm(data_paths, desc='Producing Data df'):
            temp = []
            temp.append(path)
            for name in classes:
                if re.search(name, path):
                    temp.append(int(label_info[label_info.Action == name].Ids))
            data.append(temp)
        df = pd.DataFrame(data, columns=["path", "label"])
        return df

    def split(self, len_dataset, p_train, p_test, p_val):
        """
        Parameters
        ----------
        len_dataset : TYPE
            DESCRIPTION.
        p_train : TYPE
            DESCRIPTION.
        p_test : TYPE
            DESCRIPTION.
        p_val : TYPE
            DESCRIPTION.

        Returns
        -------
        len_train : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        len_val : TYPE
            DESCRIPTION.

        """
        len_train = int(len_dataset * p_train)
        len_test = int(len_dataset * p_test)
        len_val = int(len_dataset * p_val)

        if len_dataset == len_train + len_test + len_val:
            return len_train, len_test, len_val
        else:
            difference = len_dataset - (len_train + len_test + len_val)
            return len_train, len_test + difference, len_val

    def merge_and_split(self, df, p_train=0.70, p_test=0.10, p_val=0.20, save=True):
        """
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        p_train : TYPE, optional
            DESCRIPTION. The default is 0.70.
        p_test : TYPE, optional
            DESCRIPTION. The default is 0.10.
        p_val : TYPE, optional
            DESCRIPTION. The default is 0.20
        save : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        print(
            f"SPLIT INFO: train = {p_train*100}%, validation = {p_val*100}%, test = {p_test*100}%")
        train_split, test_split, val_split = self.split(
            len(df), p_train, p_test, p_val)

        train_df = df.iloc[0:train_split]
        test_df = df.iloc[train_split:train_split + test_split]
        val_df = df.iloc[train_split + test_split:]

        if save:
            path = os.path.join(self.output_dir, 'split_info', self.comment)
            os.makedirs(path, exist_ok=True)
            train_df.to_csv(os.path.join(path, 'train.csv'))
            test_df.to_csv(os.path.join(path, 'test.csv'))
            val_df.to_csv(os.path.join(path, 'val.csv'))
            print(f"Saved split info csv files in : {path}!")
            with open(self.LOG_PATH, 'a') as f:
                with redirect_stdout(f):
                    print(
                        f"SPLIT INFO: train = {p_train*100}%, validation = {p_val*100}%, test = {p_test*100}%")
                    print(f"Saved split info csv files in : {path}!")

        return list(train_df.path), list(test_df.path), list(val_df.path)

    def get_XY(self, data_list, split="train", LSTM=False):
        """
        Parameters
        ----------
        data_list : TYPE
            DESCRIPTION.
        split : TYPE, optional
            DESCRIPTION. The default is "train".
        LSTM : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        X : TYPE
            DESCRIPTION.
        Y : TYPE
            DESCRIPTION.

        """
        X = []
        Y = []
        if not LSTM:
            for path in tqdm(data_list, desc=f"Creating X_{split} and Y_{split}"):
                data = np.array(pd.read_csv(path))
                x = data[:-1]
                y = data[-1]
                X.append(x)
                Y.append(y)
            X = np.squeeze(np.array(X))
            Y = np.squeeze(np.array(Y))
        else:
            for path in tqdm(data_list, desc=f"Creating X_{split} and Y_{split}"):
                data = np.array(pd.read_csv(path, header=None))
                x = data[:, :-1]
                y = data[0, -1]
                X.append(x)
                Y.append(y)

            X = np.array(X)
            Y = np.array(Y)

        print(f"X_{split} shape:{X.shape}")
        print(f"Y_{split} shape:{Y.shape}")
        with open(self.LOG_PATH, 'a') as f:
            with redirect_stdout(f):
                print(f"X_{split} shape:{X.shape}")
                print(f"Y_{split} shape:{Y.shape}")
        return X, Y
