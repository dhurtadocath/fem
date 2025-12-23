import os
import keras    
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from pathlib import Path
from keras import layers, initializers
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from ..utils import utils
from ..utils.path_funcs import *
# tf.compat.v1.disable_eager_execution()

# import tensorflow as tf
# import keras.utils.traceback_utils

# def dummy_error_handler(*args, **kwargs):
#     pass

# tf.keras.utils.traceback_utils.error_handler = dummy_error_handler



class SurfacePointsModelForOnePatch:
    def __init__(self, NNShape=[], diamond=False, regularizer=False, name=""):
        self.settings = {}
        if name:
            try:
                self.name = name
                
                model_path = get_abs_saved_surface_points_models_folder_path_with_model_name(name)
                # full_name = str(model_path).replace("\\", " ")
                # full_name =  full_name.replace("/", " ")
                # full_name =  full_name.split(" ")
                # full_name = str(full_name[-1])
                
                # print("Model path: ", model_path)
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
        
        if NNShape:
                total_layers = [layers.Input(shape=(3,))]#, layers.Normalization()]
                for num_of_neurons in NNShape:
                    total_layers.append(layers.Dense(num_of_neurons, activation='relu'))
                
                if regularizer:
                    total_layers.append(layers.Dense(2, activation='linear', kernel_regularizer='l2'))
                else:
                    total_layers.append(layers.Dense(2, activation='linear'))
                
                self.model = keras.Sequential(total_layers)
                self.history = []
                if diamond:
                    shape_str = ""
                    for i in NNShape:
                        shape_str = shape_str + str(i) + "-"
                    
                    self.settings = {"shape":f"{shape_str}"}

                else:
                    self.settings = {"shape":f"{num_of_neurons}-"*len(NNShape)}

    def compile(self,opt, loss_, metrics_):
        self.settings['optimizer'] = opt._name
        self.settings['loss'] = loss_
        self.settings['metrics'] = metrics_

        self.model.compile(optimizer=opt, loss=loss_, metrics=metrics_)

    def train(self, data, epochs_=5, batch_size_=64, verbose_=1, name=None):
        X_train, X_test, Y_train, Y_test = data
        if "epoch" not in self.settings:
            self.settings["epoch"] = epochs_
        if "shuffle" not in self.settings:
            self.settings["shuffle"] = True
        if "batch_size" not in self.settings:
            self.settings["batch_size"] = batch_size_

        if name is not None:
            name = name+ "-" + self.__create_file_name()
            path_training_history = Path(os.path.join(get_surface_points_model_training_data_folder_path(),name+".csv"))
            path_to_saved_model = Path(os.path.join(get_abs_saved_models_folder_path(),name+".keras"))
            try:
                
                csv_logger = CSVLogger(path_training_history)
                model_check_point = ModelCheckpoint(path_to_saved_model)
                # lr_scheduler = LearningRateScheduler(utils.scheduler)
                self.history = self.model.fit(X_train, Y_train, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=(X_test, Y_test), verbose=verbose_, callbacks=[csv_logger, model_check_point])#, lr_scheduler])
                # re_enumerate_epochs_in_csv_file(path_training_history)
            except KeyboardInterrupt:
                print("Training of model stopped!")
                print(f"path to model: {path_to_saved_model}")
                print(f"path to model training history: {path_training_history}")
    
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
                path_to_saved_model = get_abs_saved_surface_points_models_folder_path_with_model_name(name=name)
                model_training_history = get_surface_points_model_training_data_file_abs_path_by_model_name(name=name)
                
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

    # @tf.function
    def predict(self, points,verbose='auto'):
        return self.model.predict(points,verbose=verbose)
    

    def __create_file_name(self):
        try:
            print("self.settings: ", self.settings)
            return f"shape-{self.settings['shape']}bs-{self.settings['batch_size']}"
        
        except AttributeError:
            return ""


class SurfacePointsModel:
    def __init__(self, NNShape):
        total_layers = []#, layers.Normalization(axis=None)]
        point_3d_coords_input = keras.Input(shape=(3,), name="3d-point")
        patch_num_input = keras.Input(shape=(1,), name="patch")
        total_input =  layers.concatenate([point_3d_coords_input,patch_num_input])

        # total_layers.append(total_input)
        x = total_input
        for num_of_NN in NNShape:
            x = layers.Dense(num_of_NN, activation='tanh', kernel_initializer='random_normal',
    bias_initializer='zeros')(x)
        output1 = layers.Dense(1,name="first_point",activation='linear',kernel_initializer='random_normal',
    bias_initializer='zeros')(x)
        output2 = layers.Dense(1,name="second_point",activation='linear',kernel_initializer='random_normal',
    bias_initializer='zeros')(x)
        self.model = keras.Model(inputs=[point_3d_coords_input, patch_num_input], outputs={"first_point":output1, "second_point":output2}) 
        # keras.utils.plot_model(model, "my_first_model.png")
        self.history = []
        self.settings = {}
        
    def load_model(self, model):
        self.model = model

    def compile(self,opt, loss_, metrics_=0, sample_weight=None):
        self.settings['optimizer'] = opt._name
        self.settings['loss'] = loss_
        self.settings['metrics'] = metrics_

        if sample_weight is not None:
            print("self.model = ", self.model)
            self.model.compile(optimizer=opt, loss=loss_, metrics=metrics_, weighted_metrics=["mae"])
        else:
            self.model.compile(optimizer=opt, loss=loss_, metrics=metrics_)

    def train(self, data, epochs_=5, batch_size_=64, verbose_=1, name=None, sample_weights=None):
        X_train_points,X_train_patches, X_test_points,X_test_patches, Y_train_p1, Y_train_p2, Y_test_p1,Y_test_p2 = data
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
                path_to_saved_model = get_abs_saved_surface_points_models_folder_path_with_model_name(name=name)
                model_training_history = get_surface_points_model_training_data_file_abs_path_by_model_name(name=name)
                
                csv_logger = CSVLogger(model_training_history, append=True)
                model_check_point = ModelCheckpoint(path_to_saved_model)
                try:
                    if sample_weights is not None:
                        print("using sample weights " + "-#-"*10)
                        self.history = self.model.fit({"3d-point":X_train_points,"patch": X_train_patches},{"first_point":Y_train_p1, "second_point":Y_train_p2}, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=({"3d-point":X_test_points,"patch": X_test_patches}, {"first_point":Y_test_p1, "second_point":Y_test_p2}), verbose=verbose_, callbacks=[csv_logger, model_check_point], sample_weight=sample_weights)
                    else:
                        self.history = self.model.fit({"3d-point":X_train_points,"patch": X_train_patches},{"first_point":Y_train_p1, "second_point":Y_train_p2}, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=({"3d-point":X_test_points,"patch": X_test_patches}, {"first_point":Y_test_p1, "second_point":Y_test_p2}), verbose=verbose_, callbacks=[csv_logger, model_check_point])

                except KeyboardInterrupt as error:
                    print("Training of model stopped!")
                    print(f"path to model: {path_to_saved_model}")
                    print(f"path to model training history: {model_training_history}")

                finally:
                    print("re-enumerating the epochs...")
                    utils.re_enumerate_epochs_in_csv_file(model_training_history)
    
        except NameError as err:
            print("name of model not found in training folder.")
        # self.history = self.model.fit({"3d-point":X_train_points,"patch": X_train_patches},{"first_point":Y_train_p1, "second_point":Y_train_p2},
        #                                epochs=epochs_, shuffle=True,batch_size=batch_size_,
        #                                  validation_data=({"3d-point":X_test_points,"patch": X_test_patches}, {"first_point":Y_test_p1, "second_point":Y_test_p2}),
        #                                    verbose=verbose_)
        # self.settings["epochs"] = epochs_
        # self.settings["shuffle"] = True
        # self.settings["batch_size"] = batch_size_

    def predict_1_point(self, point,patch, expected_surface_points):
        tensor_point = tf.convert_to_tensor([point])
        tensor_patch = tf.convert_to_tensor([patch])
        predicted_surface_points = self.model.predict({"3d-point": tensor_point,"patch":tensor_patch})
        print(f"expected surface points: {expected_surface_points}")
        print(f"predicted surface points: {predicted_surface_points}")
        return [expected_surface_points, predicted_surface_points]
    
    def plot(self,func='loss',name='',add_to_title='' ,ext='jpg', show=True):

        title = f"{func} over {self.settings['epochs']} epochs"
        data1 = self.history.history[func]
        data2 = self.history.history[f'val_{func}']
        lowest_point1 = (len(data1)-1, data1[-1])
        lowest_point2 = (len(data2)-1, data2[-1])


        if func != 'loss':
            title = f"Accuracy over {self.settings['epochs']} epochs"
        
        if add_to_title != '':
            title = title + "\n" + add_to_title

        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(8)
        fig.set_figwidth(15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yscale("log")
        # ax.set_ylim(bottom=1e-4,top=1e+0)
        ax.set_xticks([int(len(data1)*0.5),int(len(data1)*0.75)])

        text1 = str(lowest_point1[1])
        text2 = str(lowest_point2[1])
        ax.plot(data1, color='red', label=func+f'-(final = {text1})')
        ax.plot(data2, color='blue', label=f'val_{func}'+f'-(final = {text2})')

        # final_text = AnchoredText(,loc)

        # ax.annotate(text1,
        #         xy=lowest_point1, xycoords='axes fraction',
        #         xytext=(-1.5,-1.5), textcoords='offset points',
        #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        # ax.annotate(text2,
        #         xy=lowest_point2, xycoords='axes fraction',
        #         xytext=(-1.5,1.5), textcoords='offset points',
        #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

        ax.legend()
        ax.grid(True) 
        ax.set_title(title)
        ax.set_xlabel("epochs")
        ax.set_xticks(np.arange(0,self.settings['epochs']+1, 5))
        ax.set_ylabel(func)

        
        if name != '':
            plot_path = os.path.join(get_abs_path(get_relative_saved_plots_folder_path()),f"surface_points_model/{name}.{ext}")
            plt.savefig(plot_path)

        if show:
            plt.show()
    def save_(self, name="surface_points_model"):
        path = os.path.join(get_abs_path(get_relative_saved_models_folder_path()),f"{name}.keras")
        self.model.save(path)

    def save_training_and_validation_data(self, name="surface_points_model"):
        path = os.path.join(get_abs_path(get_relative_data_folder_path()),f"model_training_history/surface_points_model/{name}.csv")
        df = pd.DataFrame(self.history.history)
        with open(path, mode='w') as f:
            df.to_csv(f)



    def print_settings(self):
        for key, value in self.settings.items():
            print(f"{key}:{value}")


    
class Experiment:

    def __init__(self, NNShape, list_epochs, list_batch_sizes,list_optimizers):
        self.nums_layers = len(NNShape)
        self.list_num_neurons_per_layer = NNShape
        self.list_epochs = list_epochs
        self.list_batch_sizes = list_batch_sizes
        self.list_optimizers = list_optimizers

        self.combinations = self.create_combinations_of_settings()

    def create_combinations_of_settings(self):
        return list(itertools.product(self.nums_layers,
                                      self.list_num_neurons_per_layer,
                                      self.list_epochs,
                                      self.list_batch_sizes,
                                      self.list_optimizers))
    
    def run(self,data, save=False):
        X_train, X_test, Y_train, Y_test = data
        combinations_of_settings = self.create_combinations_of_settings()

        for setting_ in combinations_of_settings:

            name_ = self.create_file_name_from_settings(setting_)
            number_of_layers, number_of_neurons, epochs_, batch_size_, optimizer_ = setting_
            surface_points_model = SurfacePointsModel(number_of_layers, number_of_neurons)
            surface_points_model.compile(opt=optimizer_, loss_="mae", metrics_=['accuracy'])
            surface_points_model.train((X_train, X_test, Y_train, Y_test),epochs_, batch_size_, verbose_=1)
            surface_points_model.plot(name=name_,add_to_title="loss function: mean absolute error" ,show=False)
            
            if save:
                surface_points_model.save_(name=name_)


    def create_file_name_from_settings(self, settings):
        number_of_layers, number_of_neurons, epochs_, batches_, optimizer_ = settings
        return f"nl-{number_of_layers}-nn-{number_of_neurons}-b-{batches_}-opt-{optimizer_._name}"

    
    def __str__(self):
        return f"The Experiment will try all different combinations of the following:\n\
        Number of Layers:{self.nums_layers}\n\
        Number of neurons per layer:{self.list_num_neurons_per_layer}\n\
        Number of epcohs:{self.list_epochs}\n\
        Number of batches:{self.list_batch_sizes}\n\
        List of optimizers:{self.list_optimizers}\n"
        

