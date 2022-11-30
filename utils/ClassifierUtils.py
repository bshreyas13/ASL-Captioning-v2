# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:25:28 2022

@author: sbhat
"""
import os
import sys
import json
import numpy as np 
import yaml
from contextlib import redirect_stdout
import seaborn as sns
from sklearn import svm 
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, Adadelta,RMSprop, SGD, Adamax
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
#from tensorflow_addons.optimizers import AdamW
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import precision_recall_fscore_support as scorer

class GestureClassifier():
    def __init__(self,train_ds,test_ds, val_ds=None, label_info =None, 
                 save_path=None, output_dir=os.getcwd(),comment='_'):
        """
        Parameters
        ----------
        train_ds : tuple
            Tuple of arrays (X_train,Y_train).
        test_ds : tuple
            Tuple of arrays (X_test,Y_test).
        val_ds : tuple, optional
                Tuple of arrays (X_val,Y_val). The default is None.
        label_info : dataframe, optional
            DataFrame maping Actions --> Categorical Label --> Binary Label. The default is None.
        output_dir : str, path, optional
            Path to output working directory. The default is os.getcwd().
        comment : str, optional
            Comment to save files with unique name. The default is "_".
        
        Returns
        -------
        None.

        """
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.val_ds = val_ds
        self.label_info = label_info
        self.output_dir = output_dir
        self.save_path = save_path
        self.comment = comment
        self.LOG_PATH = os.path.join(self.output_dir,f'RUN_LOG_{comment}.txt')
    
    ## BUILD SVM 
    def SVM_Pipeline(self, kernel = 'rbf',C=1.0,gamma=0.5,degree=3):
        """
        Parameters
        ----------
        kernel : str, optional
            SVM kernel to use. The default is 'rbf'.
        C : float, optional
            C for rbf kernel. The default is 1.0.
        gamma : float, optional
            gamma for rbf kernel. The default is 0.5.
        degree : int, optional
            degree of polynomial kernel. The default is 3.

        Returns
        -------
        clf : sklearn SVM object
            Trained SVM for classification.

        """
        X_train = self.train_ds[0]
        Y_train = self.train_ds[1]
        if kernel == 'rbf':
            clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
        elif kernel == 'poly':
            clf = svm.SVC(kernel='poly', degree=degree, C=C)
        elif kernel == 'linear':
            clf = svm.SVC(kernel='linear')
        clf.fit(X_train, Y_train)
        
        return clf
    
    def make_predictions_SVM(self,clf):
        """    
        Parameters
        ----------
        clf : sklearn.SVM object
            trained_SVM classifier.

        Returns
        -------
        predictions : float
            float value of predicted class.

        """
        X_test = self.test_ds[0]
        predictions = clf.predict(X_test)
        return predictions
    
    def train_SVM(self, max_C = 10001 , C_steps = 100, kernels=['poly','rbf']):
        """
        Parameters
        ----------
        max_C : TYPE, optional
            DESCRIPTION. The default is 10001.
        C_steps : TYPE, optional
            DESCRIPTION. The default is 100.
        kernels : TYPE, optional
            DESCRIPTION. The default is ['poly','rbf'].

        Returns
        -------
        None.

        """
        
        perf_metrics = {'accuracy':[],'f1-score':[],'kernel':[]}
        
        for kernel in kernels:
            if kernel == 'rbf':
                for C in range(1,max_C,C_steps):
                    print(f"Training and testing SVM with {kernel} kernel and C={C}")
                    
                    classifier = self.SVM_Pipeline(kernel=kernel,C=C)
                    y_preds = self.make_predictions_SVM(classifier)
                    accuracy = accuracy_score(self.test_ds[1], y_preds)*100
                    f1 = f1_score(self.test_ds[1], y_preds, average='weighted')*100
                    print(f'Metric Scores with kernel={kernel} and C={C}: Accuracy:{accuracy}%, f1:{f1}%')
                    perf_metrics['accuracy'].append(accuracy)
                    perf_metrics['f1-score'].append(f1)
                    perf_metrics['kernel'].append(kernel)
                
                ## Save performance metrics as json file
                with open(os.path.join(self.output_dir,"performance_metrics_rbf.json"), "w") as outfile:
                    json.dump(perf_metrics, outfile)
            

            elif kernel == 'poly':
                for degree in range(3,6):
                    print(f"Training and testing SVM with {kernel} kernel and degree={degree}")       
                    classifier = self.SVM_Pipeline(kernel=kernel,degree=degree)
                    y_preds = self.make_predictions_SVM(classifier)
                    accuracy = accuracy_score(self.test_ds[1], y_preds)*100
                    f1 = f1_score(self.test_ds[1], y_preds, average='weighted')*100
                    print(f'Metric Scores with kernel={kernel} and degree={degree}: Accuracy:{accuracy}%, f1:{f1}%')
                    perf_metrics['accuracy'].append(accuracy)
                    perf_metrics['f1-score'].append(f1)
                    perf_metrics['kernel'].append(kernel)
                
                ## Save performance metrics as json file
                with open(os.path.join(self.output_dir,"performance_metrics_poly.json"), "w") as outfile:
                    json.dump(perf_metrics, outfile)
            
            else:
                print("Implmentation unailable. Only use kernels = ['rbf','poly'], or any one of them in a list.")
                sys.exit()
    
    ## BUILD NEURAL NET
    def NN_pipeline(self,model_name='LSTM',n_layers=2,hidden_units = 100, mode='train',
                    hyperparamters={'optimizer':'Adam','epochs':100,'batch_size':10},
                    saved_model=None,INCLUDE_CONV=True):
        """
        Parameters
        ----------
        model_name : str, optional
            LSTM or MLP. The default is 'MLP'.
        n_layers : int, optional
            Depth of the network. The default is 2.
        hidden_units : int, optional
            Number of hidden units, breadth of the network
        mode : str, optional
            Pipeline mode train, test, or infer. The default is 'train'.
        hyperparamters : dict, optional
            Dict of hyperparameters for model training. The default is {'optimizer':'Adam','epochs':100,'batch_size':10}.
        saved_model : model.h5 (tensorflow), optional
            Pretrained model if available. Not optional for test or infer mode. The default is None.
        INCLUDE_CONV : Bool, optional
            True if conv1D layer is to be used. The default is True.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        """
        ## SETUP
        in_shape = (self.train_ds[0].shape[-1],)
        num_classes = len(np.unique(self.train_ds[1]))
        optimizer = self.get_optimizer(hyperparamters['optimizer'])
        epochs = hyperparamters['epochs']
        batch_size = hyperparamters['batch_size']
        
        ## GET MODEL
        if model_name == 'MLP':
            model = self.MLP(in_shape,num_classes,n_layers)
            model.summary()
        if model_name == 'LSTM':
            in_shape = (self.train_ds[0].shape[1],self.train_ds[0].shape[2])
            model = self.LSTM(in_shape=in_shape,num_classes=num_classes,
                              n_layers=n_layers, units=hidden_units, INCLUDE_CONV=INCLUDE_CONV)
            model.summary()
        else:
            print("\nWarning: Exiting program. Use one of the 2:MLP. LSTM")
            with open(self.LOG_PATH, 'w') as f:
                with redirect_stdout(f):
                    print("\nWarning: Exiting program. Use one of the 2:MLP. LSTM")
            sys.exit()
        
        ## TRAIN/TEST
        if mode == 'train':
            train_history, trained_model = self.train_net(model, optimizer, epochs, batch_size, model_name)
            score = self.test_net(trained_model, batch_size)
            train_record = Utils(self.output_dir)
            train_record.plot_metrics(train_history, score, model_name,self.comment)
        
        if mode == 'test':
            assert saved_model != None
            # load saved_model
            trained_model = load_model(saved_model)
            score  = self.test_net(trained_model, batch_size)
            with open(self.save_path, 'w') as f:
                with redirect_stdout(f):
                    model.summary()
                    
            return model
            
    
    def MLP(self,in_shape, num_classes, n_layers=2):
        """
        Parameters
        ----------
        in_shape : tuple
            (n_features) as a tuple to set input layer shape.
        num_classes : int
            Number of output classes.
        n_layers : int, optional
            Depth of the network. The default is 2.

        Returns
        -------
        model : tf.keras.Model object
            Untrained tensorflow model

        """
        ## First Dense layer hidden_units
        units = n_layers * 10
        
        inputs = keras.Input(shape=in_shape)
        x = inputs
        for layer_ in range(n_layers):    
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(units,activation='relu')(x)
            units /= 2
        outputs = layers.Dense(num_classes,activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="GC_MLP")
        
        ## plot model 
        self.save_path = os.path.join(self.output_dir,'GC_MLP.txt')
        ## Not using plot model as it can potentially throw error about graphviz
        # keras.utils.plot_model(model,
        #                        to_file=self.save_path,
        #                        show_shapes=True,
        #                        dpi=96,
        #                        show_layer_activations=True)
        with open(self.save_path, 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model
    
    def LSTM(self,in_shape, num_classes, n_layers=1, units = 100, INCLUDE_CONV=True):
        """
        Parameters
        ----------
        in_shape : tuple
            (timesteps, n_features) as a tuple to set input layer shape.
        num_classes : int
            Number of output classes.
        n_layers : int, optional
            Number of conv1D->dropout->LSTM->dropout layers in architeture, depth of the network. The default is 1.
        units : int, optional
            Number of hidden units in LSTM, breadth of the network. The default is 100.
        INCLUDE_CONV : Bool, optional
            True if conv1D layer is to be used. The default is True.

        Returns
        -------
        model : tf.keras.Model object
            Untrained tensorflow model.

        """
        inputs = keras.Input(shape=in_shape)
        x = inputs
        
        for layer_ in range(n_layers):
            if INCLUDE_CONV:
                x = layers.Conv1D(filters=32,kernel_size=3,padding='same',activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            if layer_ == n_layers-1:    
                x = layers.LSTM(units,dropout=0.2,recurrent_dropout=0.1)(x)
                x = layers.Dropout(0.2)(x)
            else:
                x = layers.LSTM(units,dropout=0.2,recurrent_dropout=0.1,return_sequences=True)(x)
                units = int(units/2)
                x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes,activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="GC_LSTM")
        
        ## plot model 
        self.save_path = os.path.join(self.output_dir,f'GC_LSTM_{self.comment}.txt')
        ## Not using plot model as it can potentially throw error about graphviz
        # keras.utils.plot_model(model,
        #                        to_file=self.save_path,
        #                        show_shapes=True,
        #                        dpi=96,
        #                        show_layer_activations=True)
        with open(self.save_path, 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model
    
    ##Learning Rate Schedule ##
    def lr_schedule(self,epoch):
        """
        Parameters
        ----------
        epoch : int
            This is passed by lr_scheduler to update lr with epoch.

        Returns
        -------
        lr : float
            Learning Rate.

        """
    
        lr = 1e-3
        if epoch > 200:
            lr *= 0.5e-3
        elif epoch > 180:
            lr *= 1e-3
        elif epoch > 160:
            lr *= 1e-2
        elif epoch > 120:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr
    
    def train_net(self,model, optimizer,epochs,batch_size, model_name):
        """
        Parameters
        ----------
        model : tf.keras model
            model to train.
        optimizer : tf.keras optimizer
            optimizer to use with model.compile.
        epochs : int
            Number of epochs of training.
        batch_size : int
            batch size for training.
        model_name : str
            MLP/LSTM.

        Returns
        -------
        history : tf.keras history
            Training history.

        """
        x_train, y_train = self.train_ds
        x_val, y_val = self.val_ds
        
        encoder =  Utils(self.output_dir)
        y_train = encoder.get_binary_label(y_train, self.label_info)
        y_val = encoder.get_binary_label(y_val, self.label_info)
        # Compile model
        model.compile(loss= 'categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        # prepare model model saving directory.
        save_dir = os.path.join(self.output_dir, 'trained_models')
        os.makedirs(save_dir,exist_ok=True)
        save_name = f'GC_{model_name}'+'_{epoch:02}.h5' 
        filepath = os.path.join(save_dir, model_name+f'_{self.comment}',save_name)
      
        # prepare callbacks for model saving and for learning rate adjustment. 
        steps_per_epoch=len(x_train)//batch_size
        save_period = 10
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_freq=int(steps_per_epoch*save_period))
      
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
      
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                     cooldown=0,
                                     patience=5,
                                     min_lr=0.5e-6)
      
        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        # Train the model 
        history=model.fit(x_train, y_train, batch_size=batch_size, 
                          epochs=epochs,validation_data= (x_val,y_val),
                          callbacks=callbacks)
      
        # trained_model = tf.keras.models.load_model(os.path.join(save_dir,save_name))
        return history, model
    
    def test_net(self,model,batch_size):
        """
        Parameters
        ----------
        model : tf.keras model object
            Keras model to test.
        batch_size : int
            Data batch size.

        Returns
        -------
        score : float
            Test accuracy.

        """
        
        x_test, y_test = self.test_ds
        encoder =  Utils(self.output_dir)
        y_test = encoder.get_binary_label(y_test, self.label_info)
        # Evaluate Model on Test set
        score = model.evaluate(x_test,
                             y_test,
                             batch_size=batch_size,
                             verbose=2)
        print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
        with open(self.LOG_PATH, 'a') as f:
            with redirect_stdout(f):
                print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
        return score[1]*100
    
    def get_optimizer(self,name):
        """
        Parameters
        ----------
        name : str
            Optimizer name, use one of the following;{Adam, SGD, Adamax, Adadelta, RMSprop}.

        Returns
        -------
        Keras Optimizer Object
            Otimizer seclected.

        """
        if name == 'Adam':
            return Adam(learning_rate = self.lr_schedule(0))
        # elif name == 'AdamW':
        #     wd = lambda: 1e-3 * lr_schedule(0)
        #     return AdamW(learning_rate = self.lr_schedule(0),weight_decay=wd)
        elif name == 'SGD':
            return SGD(learning_rate = self.lr_schedule(0))
        elif name == 'Adamax':
            return Adamax(learning_rate = self.lr_schedule(0))
        elif name == 'Adadelta':
            return Adadelta(learning_rate = self.lr_schedule(0))
        elif name == 'RMSprop':
            return RMSprop(learning_rate = self.lr_schedule(0))
        else:
            print("\nWarning: Exiting program. Use one of the 6: Adamw, Adam, Adamax, Adadelta, SGD, RMSprop")
            with open(self.LOG_PATH, 'a') as f:
                with redirect_stdout(f):
                    print("\nWarning: Exiting program. Use one of the 6: Adamw, Adam, Adamax, Adadelta, SGD, RMSprop")
            sys.exit()

        
class Utils():
    
    def __init__(self,output_dir,raw_dataset=False):
        """   
        Parameters
        ----------
        output_dir : str, path
            Filepath to save plots.

        Returns
        -------
        None.

        """
        self.output_dir = output_dir
        self.raw_dataset = raw_dataset
        
    def plot_metrics(self,history,score,model_name,comment='_'):
        """
        Parameters
        ----------
        history : tf history object
            tf2 history dictionary with train, val metrics.
        score : float
            test accuracy.
        model_name : str
            String model name.

        Returns
        -------
        None.

        """
        #Plot training curve
        sns.set()
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.plot(history.history['accuracy'])
        ax.plot(history.history['val_accuracy'])
        ax.text(1,1, f'Test accuracy:{score}%',fontweight='bold',
                 size =20, bbox ={'facecolor':'Green','alpha':0.5, 'pad':10})
        ax.set_title('model accuracy')
        ax.set_ylabel('accuracy')
        ax.set_xlabel('epoch')
        ax.legend(['train', 'val'], loc='upper left')
        fig.savefig(os.path.join(self.output_dir,model_name+comment+'.png'))
        plt.close()
        
    def get_binary_dict(self,label_info):
        """
        Parameters
        ----------
        label_info : Dataframe
            Pandas dataframe with label, Ids and Binary Label.

        Returns
        -------
        label_bins : dict
            Dict mapping categorical label to One Hot encoded labels.
        bins_to_action
            Dict mapping One Hot encoded labels to Action.
        """
        label_bins = {}
        bins_to_action = {}
        for ids in list(label_info.Ids):
            label_chars = list(np.array(label_info[label_info.Ids==ids].binary_label).flatten()[0])
            a_label = []
            for char in label_chars:
                if char.isdigit():
                    a_label.append(int(char))
            a_label = np.array(a_label)
            label_bins[ids]=a_label
            bins_to_action[str(a_label)]= label_info[label_info.Ids==ids].Action.to_string().split(' ')[-1]
        return label_bins, bins_to_action
    
    def get_binary_label(self,Y,label_info):
        """
        Parameters
        ----------
        Y : array
            Array of Labels (n_samples,).
        label_info : Dataframe
            Pandas dataframe with label, Ids and Binary Label.

        Returns
        -------
        out_Y : array
            Array of One Hot Encoded Labels (n_samples,4).

        """
        bin_dict,_ = self.get_binary_dict(label_info)
        out_Y = []
        for elem in list(Y) :
            bin_label = bin_dict[int(elem)]
            out_Y.append(bin_label)
        out_Y = np.array(out_Y)
        return out_Y
    
    def load_model_configs(self,path):
        with open(path) as out:
            configs = yaml.load(out, Loader=yaml.FullLoader)
        ## RAW data params
        classes = configs["CLASSES"]
        ## POSE MACHINE params
        body_pose = configs['BODY_POSE']
        high_res_hands = configs['HIGH_RES_HANDS']
        augment = configs['AUGMENT_DATA']
        ## MODEL Parameters
        model_name = configs["MODEL"]
        mode = configs["MODE"]
        n_layers = configs["NUM_LAYERS"]
        n_units = configs["NUM_UNITS"]
        optimizer = configs["OPTIMIZER"]
        epochs = configs["EPOCHS"]
        batch_size = configs["BATCH_SIZE"]
        saved_model = configs["SAVED_MODEL_PATH"]
        INCLUDE_CONV = configs["INCLUDE_CONV_LAYER"]
        ## Data parameters
        wsize = configs["WINDOW_SIZE"]
        stride = configs["STRIDE"]   
        ## Inference Params
        in_mode = configs['INPUT_MODE']
        save_video = configs['SAVE_VIDEO']
        play_video = configs['PLAY_VIDEO']
        
        if mode == 'train':
            comment = f"layers{n_layers}_units{n_units}_{optimizer}_bs{batch_size}_conv{INCLUDE_CONV}_wsize{wsize}_stride{stride}"
        elif mode == 'test':
            comment = f"{model_name}_{mode}"
        elif mode == 'inference':
            comment = f"{model_name}_{mode}"
        else:
            print(f"Choose mode correctly: train/test/inference. Mode selected {mode}")
            sys.exit()
        
        if not self.raw_dataset:
            print("MODEL CONFIGS")
            print("==============")
            LOG_PATH = os.path.join(self.output_dir,f'RUN_LOG_{comment}.txt')
            ## LOG
            with open(LOG_PATH, 'w') as f:
                with redirect_stdout(f):
                    print("MODEL CONFIGS")
                    print("==============")
            ## Print on console
            for key, value in configs.items():
                print(key, ": ", value)
                with open(LOG_PATH, 'a') as f:
                    with redirect_stdout(f):
                        print(key, ": ", value)
        ## Return
        if self.raw_dataset:
            return [body_pose, high_res_hands, classes, augment]
        
        elif mode == 'train' or mode == 'test':         
            return [body_pose, high_res_hands, model_name,mode,
                    n_layers,n_units,optimizer,epochs, batch_size,
                    saved_model,INCLUDE_CONV,wsize,stride,comment]
      
        elif mode == 'inference':      
            return [body_pose, high_res_hands, model_name,mode,
                    saved_model,wsize,stride,comment,
                    in_mode,save_video,play_video]
