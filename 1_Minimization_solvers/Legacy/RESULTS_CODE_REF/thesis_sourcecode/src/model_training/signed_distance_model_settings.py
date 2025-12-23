import os
import keras
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
from keras.callbacks import CSVLogger, ModelCheckpoint
# from keras.layers.activation import LeakyReLU
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
from ..utils import utils
from ..utils.path_funcs import *
# from ..data_processing.data_organized import 
# tf.compat.v1.disable_eager_execution()


# import tensorflow as tf
# import keras.utils.traceback_utils

# def dummy_error_handler(*args, **kwargs):
#     pass

# tf.keras.utils.traceback_utils.error_handler = dummy_error_handler


class SignedDistanceModel:
    def __init__(self, NNShape=[], diamond=False, regularizer=False, name=""):
        self.settings = {}
        if name:
            try:
                self.name = name
                
                model_path = get_abs_saved_signed_distance_models_folder_path_with_model_name(name)
                # full_name = str(model_path).replace("\\", " ")
                # full_name =  full_name.replace("/", " ")
                # full_name =  full_name.split(" ")
                # full_name = str(full_name[-1])
                
                print("Model path: ", model_path)
                # print("full_name: ", full_name)
                # training_history_csv_path = get_patch_model_training_data_file_abs_path_by_model_name(name)
                self.model = keras.models.load_model(model_path)
                # colnames = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
                # self.history = pd.read_csv(training_history_csv_path,names=colnames, header=0)
                self.settings["shape"] = utils.get_shape_from_name(str(model_path))
                self.settings["shape_as_list"] = utils.get_shape_from_name_as_list(str(model_path))
                self.settings["batch_size"] = utils.get_batch_size_from_name(str(model_path))
                
            except FileNotFoundError as error:
                print(f"{error}")
        
        else:
            
            if NNShape:
                total_layers = [layers.Input(shape=(3,))]
                for num_of_neurons in NNShape:
                    total_layers.append(layers.Dense(num_of_neurons, activation="relu"))
                    # total_layers.append(LeakyReLU(alpha=0.01))
                
                if regularizer:
                    total_layers.append(layers.Dense(1, activation='linear', kernel_regularizer='l2'))
                else:
                    total_layers.append(layers.Dense(1, activation='linear'))
                
                self.model = keras.Sequential(total_layers)
                self.history = []
                if diamond:
                    shape_str = ""
                    for i in NNShape:
                        shape_str = shape_str + str(i) + "-"
                    
                    self.settings = {"shape":f"{shape_str}"}

                else:
                    self.settings = {"shape":f"{num_of_neurons}-"*len(NNShape)}


    def compile(self,opt, loss_, metrics_, sample_weight=False):
        self.settings['optimizer'] = opt._name
        self.settings['loss'] = loss_
        self.settings['metrics'] = metrics_

        if sample_weight:
            print("self.model = ", self.model)
            self.model.compile(optimizer=opt, loss=loss_, metrics=metrics_, weighted_metrics=["accuracy"])
        else:
            self.model.compile(optimizer=opt, loss=loss_, metrics=metrics_)

    def retrain_existing_model(self, data, epochs_=5, batch_size_=64, verbose_=1, name=None, sample_weights=None):
        
        X_train, X_test, Y_train, Y_test = data
        if "epoch" not in self.settings:
            self.settings["epoch"] = epochs_
        if "shuffle" not in self.settings:
            self.settings["shuffle"] = True
        if "batch_size" not in self.settings:
            self.settings["batch_size"] = batch_size_

        try:
            if name is None:
                raise NameError
            else:
                path_to_saved_model = get_abs_saved_signed_distance_models_folder_path_with_model_name(name=name)
                model_training_history = get_signed_distance_model_training_data_file_abs_path_by_model_name(name=name)
                
                csv_logger = CSVLogger(model_training_history, append=True)
                model_check_point = ModelCheckpoint(path_to_saved_model)
                try:
                    if sample_weights is not None:
                        print("using sample weights " + "-#-"*10)
                        self.history = self.model.fit(X_train, Y_train, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=(X_test, Y_test), verbose=verbose_, callbacks=[csv_logger, model_check_point], sample_weight=sample_weights)
                    else:
                        self.history = self.model.fit(X_train, Y_train, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=(X_test, Y_test), verbose=verbose_, callbacks=[csv_logger, model_check_point])

                except KeyboardInterrupt as error:
                    print("Training of model stopped!")
                    print(f"path to model: {path_to_saved_model}")
                    print(f"path to model training history: {model_training_history}")

                finally:
                    print("re-enumerating the epochs...")
                    utils.re_enumerate_epochs_in_csv_file(model_training_history)
    
        except NameError as err:
            print("name of model not found in training folder.")
        
    def train(self, data, epochs_=5, batch_size_=64, verbose_=1, name=None, sample_weights=None):
        X_train, X_test, Y_train, Y_test = data
        if "epoch" not in self.settings:
            self.settings["epoch"] = epochs_
        if "shuffle" not in self.settings:
            self.settings["shuffle"] = True
        if "batch_size" not in self.settings:
            self.settings["batch_size"] = batch_size_

        if name is not None:
            name = name+ "-" + self.__create_file_name()
            path_training_history = Path(os.path.join(get_signed_distance_model_training_data_folder_path(),name+".csv"))
            path_to_saved_model = Path(os.path.join(get_abs_saved_models_folder_path(),name+".keras"))
            try:
                
                csv_logger = CSVLogger(path_training_history)
                model_check_point = ModelCheckpoint(path_to_saved_model)
                if sample_weights is not None:
                    print("using sample weights " + "-#-"*10)
                    self.history = self.model.fit(X_train, Y_train, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=(X_test, Y_test), verbose=verbose_, callbacks=[csv_logger, model_check_point], sample_weight=sample_weights)
                else:
                    self.history = self.model.fit(X_train, Y_train, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=(X_test, Y_test), verbose=verbose_, callbacks=[csv_logger, model_check_point])
                # re_enumerate_epochs_in_csv_file(path_training_history)
            except KeyboardInterrupt:
                print("Training of model stopped!")
                print(f"path to model: {path_to_saved_model}")
                print(f"path to model training history: {path_training_history}")
            

    def add_L2_regularizer_to_output_layer(self):
        last_layer = self.model.get_layer(index=-1)
        last_layer_weights = last_layer.get_weights()
        new_last_layer_with_l2_regularizer = layers.Dense(96, activation='softmax', kernel_regularizer='l2', name="output")
        # new_last_layer_with_l2_regularizer.set_weights(last_layer_weights)
        # print("###model summary before :", )
        # self.model.summary()
        # print("###model summary after pop: ")
        self.model.pop()
        # self.model.summary()
        # print("##model summary after add: ")
        self.model.add(new_last_layer_with_l2_regularizer)
        self.model.layers[-1].set_weights(last_layer_weights)
        # self.model.summary()



        
    def predict_1_point(self, point, expected_signed_distance):
        tensor_point = tf.convert_to_tensor([point])
        predicted_signed_distance = np.argmax(self.model.predict(tensor_point))
        print(f"expected expected_signed_distance: {expected_signed_distance}")
        print(f"predicted expected_signed_distance: {predicted_signed_distance}")
        return [expected_signed_distance, predicted_signed_distance]

    # def predict(self, points):
    #     predictions = self.model.predict(points)
    #     predicted_signed_distances = [np.argmax(i) for i in predictions]
    #     return predicted_signed_distances
    
    # @tf.function
    def predict(self, points):
        # points=tf.convert_to_tensor(points)
        return self.model.predict(points,verbose=0)
    
    def __create_file_name(self):
        try:
            print("self.settings: ", self.settings)
            return f"shape-{self.settings['shape']}bs-{self.settings['batch_size']}"
        
        except AttributeError:
            return ""