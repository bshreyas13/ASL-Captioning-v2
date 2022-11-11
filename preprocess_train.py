# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:30:27 2022

@author: sbhat
"""

import os
import argparse
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score
from contextlib import redirect_stdout
import glob
import traceback
import shutil
##CUSTOM MODULES
from utils.DatasetLab import DatasetUtils
from utils.ClassifierUtils import GestureClassifier
from utils.ClassifierUtils import Utils

## Method to parse args from commnad line 
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", help="path to configs directory.", default=f"{os.path.join(os.getcwd(),'configs')}")
    args.add_argument("--data_dir", "-d", help="data directory ", default=f"{os.path.join(os.path.dirname(os.getcwd()),'WORKDIR','RAW_Dataset')}") 
    args.add_argument("--out_dir", "-o", help="output directory ", default=f"{os.path.join(os.path.dirname(os.getcwd()),'WORKDIR')}")  
    args = args.parse_args()
    return args.config, args.data_dir, args.out_dir

def load_dataset(tool,classes,label_info,LSTM=False):
    """
    Parameters
    ----------
    tool : DatasetLab object
        Dataset Label object with paths initialized.
    classes : list
        List of string class labels.
    label_info : DataFrame
        Pandas dataframe with label, Ids and Binary Label.

    Returns
    -------
    train_ds : Tuple
        Tuple of X_train,Y_train.
    test_ds : Tuple
        Tuple of X_test, Y_test.
    val_ds : Tuple
        Tuple of X_val , Y_val.

    """
    data_paths = tool.get_path_list()
    data_df = tool.get_df(data_paths, classes, label_info)
    train_list, test_list, val_ds =  tool.merge_and_split(data_df)  
    train_ds = tool.get_XY(train_list,split="train",LSTM=LSTM)
    test_ds = tool.get_XY(test_list,split="test",LSTM=LSTM)
    val_ds = tool.get_XY(test_list,split="val",LSTM=LSTM)
    return train_ds,test_ds,val_ds

def check_data(class_dirs,out_dir):
    """
    Parameters
    ----------
    class_dirs : list
        List of paths to class directories.
    out_dir : str, path
        Path to output workdir.

    Returns
    -------
    process_raw : Bool
        Bool flag to skip process raw dataset step. False if skip.
    resample : Bool
        Bool flag to skip  resample dataset step. False if skip.

    """
    ## CHECK : Processed data by class exists, process_raw==False 
    process_raw = False
    resample = True
    for directory in class_dirs: 
        if len(os.listdir(directory)) == 0:
            process_raw = True
    if 'resample_state.txt' in os.listdir(out_dir):
        with open(os.path.join(out_dir,"resample_state.txt"),"r") as file:
            lines = file.readlines()
        if lines !=0:
            w_size = float(lines[0].split(':')[1].split(';')[0])
            stride_ = float(lines[0].split(':')[2].split(' ')[0])
            if w_size == win_size and stride_==stride:
                resample = False
            else:
                shutil.rmtree(tool.dataset_dir)
                tool.prepare_directories(classes)
                process_raw=True
            return process_raw, resample 
        else:
            print("Error with resampled data. Process Raw Data and try again!")
            sys.exit()       
    
    if not resample:
        print(f"{lines[1]}!\nParameters used --> {lines[0]}")
    
    return process_raw, resample

if __name__=="__main__":
    
    config_dir, in_dir , out_dir = parse_args()
    os.makedirs(out_dir,exist_ok=True)
    
    config_tool = Utils(out_dir)
    
    ## LOAD: Configs
    configs = os.path.join(config_dir,'*.yaml')
    for config_path in glob.iglob(configs, recursive=True):
        all_configs = config_tool.load_model_configs(config_path)
        _,_,model_name,mode,n_layers,n_units,optimizer,epochs,batch_size,saved_model,INCLUDE_CONV,win_size,stride,comment = all_configs
        tool = DatasetUtils(in_dir, out_dir,comment=comment)
        LOG_PATH = os.path.join(out_dir,f'RUN_LOG_{comment}.txt')
        
        ## SETUP: read raw data info files and extract info needed
        video_info = tool.read_annotations_df(os.path.join(os.path.dirname(in_dir),'video_info.csv'))
        label_info = tool.read_annotations_df(os.path.join(os.path.dirname(in_dir),'label_info.csv'))
        
        classes = list(label_info.Action)
        class_dirs = tool.prepare_directories(classes)
        fps = int(list(video_info.fps)[0])
        
        process_raw , resample = check_data(class_dirs, out_dir)
     
        ## PROCESS RAW DATA VECTOR: Produce keypoints:labels by class from videos
        if process_raw:
            for vec in tqdm(os.listdir(in_dir), desc="Processing raw data vectors"):
                vector = tool.load_vector(os.path.join(in_dir,vec))        
                vectors_by_label = tool.get_vectors_by_label(vector)
                for label,vector in vectors_by_label.items():
                    name_label = list(label_info[label_info.Ids == label].Action)[0]
                    save_dir = os.path.join(tool.dataset_dir,name_label)
                    tool.save_vector(vector,save_dir, vec)
        else:
            print("Raw Data already processed. Resampling and reshaping.")
            
        ## READ VECTORS : Read raw vectors for each class and reshape vectors
        ## Resample with window size frames >> X.shape(n_samples,wsize*63). 
        ## Label and shuffle the new data
        if resample:
            for directory in tqdm(class_dirs,desc="Resampling and Reshaping data"):
                for file in os.listdir(directory):
                 
                    data = tool.load_vector(os.path.join(directory,file))
                    save_dir = os.path.join(tool.dataset_dir,directory)
                    tool.resample_reshape_data(win_size, stride, fps, data, 
                                               save_dir, file,LSTM = True, INCLUDE_FRAME_NO=False)
            ## write file to note if resampling completed successfully
            file = open(os.path.join(out_dir,"resample_state.txt"),"w")
            file.write(f"Window_size:{win_size};Stride:{stride} seconds\nData already resampled and saved")
            file.close()
    
        if model_name == 'SVM':
            ## READ RESAMPLED VECTORS: Read the resampled vectors,get X,Y    
            train_ds,test_ds,val_ds = load_dataset(tool, classes, label_info, LSTM=False)
            clf_tool = GestureClassifier(train_ds, test_ds, output_dir=out_dir)
            max_C = 1000 + 1
            C_steps = 100
            kernels = ['poly','rbf'] 
            ##TRAIN SVM : Train SVM for classifier
            clf_tool.train_SVM( max_C = max_C, C_steps = C_steps, kernels=kernels)
        
        elif model_name == 'MLP':
            try:
                ## READ RESAMPLED VECTORS: Read the resampled vectors,get X,Y    
                train_ds,test_ds,val_ds = load_dataset(tool, classes, label_info, LSTM=False)
                clf_tool = GestureClassifier(train_ds, test_ds, val_ds, label_info, output_dir=out_dir)
                hyperparamters={'optimizer':optimizer,'epochs':epochs,'batch_size':batch_size}
                ## TRAIN MLP
                clf_tool.NN_pipeline(model_name=model_name,n_layers=n_layers,mode=mode,
                                 hyperparamters=hyperparamters)
            except Exception as ex:
                traceback.print_exc()
                with open(LOG_PATH, 'a') as f:
                    with redirect_stdout(f):
                        print("ERROR!Caught Exception:",ex)
                        print(traceback.format_exc())
                        
        elif model_name == 'LSTM':
            try:
                ## READ RESAMPLED VECTORS: Read the resampled vectors,get X,Y    
                train_ds,test_ds,val_ds = load_dataset(tool, classes, label_info, LSTM=True)
                clf_tool = GestureClassifier(train_ds, test_ds, val_ds, label_info, output_dir=out_dir,comment=comment)
                hyperparamters={'optimizer':optimizer,'epochs':epochs,'batch_size':batch_size}
                ## TRAIN LSTM
                clf_tool.NN_pipeline(model_name=model_name,n_layers=n_layers, hidden_units=n_units,mode=mode,
                                      hyperparamters=hyperparamters, saved_model=saved_model,INCLUDE_CONV = INCLUDE_CONV)           
            except Exception as ex:
                traceback.print_exc()
                with open(LOG_PATH, 'a') as f:
                    with redirect_stdout(f):
                        print("ERROR!Caught Exception:",ex)
                        print(traceback.format_exc())
                        