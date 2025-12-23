import os
import keras
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
# from keras.layers.activation import LeakyReLU
from tensorflow.keras.layers import LeakyReLU
from keras import initializers
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
from ..utils import utils
from ..utils.path_funcs import *
from ..data_processing.data_organized import collect_csv_files_into_one_df, get_training_and_testing_data_for_patch_model
from pdb import set_trace
# tf.compat.v1.disable_eager_execution()

# import tensorflow as tf
# import keras.utils.traceback_utils

# def dummy_error_handler(*args, **kwargs):
#     pass

# tf.keras.utils.traceback_utils.error_handler = dummy_error_handler

class PatchClassificationModel:
    def __init__(self, NNShape=[], diamond=False, regularizer=False, name=""):
        self.settings = {}
        if name:
            try:
                self.name = name
                
                model_path = get_abs_saved_patch_models_folder_path_with_model_name(name)
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
            # initializer = initializers.GlorotNormal()
            if NNShape:
                total_layers = [layers.Input(shape=(3,))]
                for num_of_neurons in NNShape:
                    total_layers.append(layers.Dense(num_of_neurons, activation="relu"))
                    # total_layers.append(LeakyReLU(alpha=0.01))
                
                if regularizer:
                    total_layers.append(layers.Dense(96, activation='softmax', kernel_regularizer='l2'))
                else:
                    total_layers.append(layers.Dense(96, activation='softmax'))
                
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
                path_to_saved_model = get_abs_saved_patch_models_folder_path_with_model_name(name=name)
                model_training_history = get_patch_model_training_data_file_abs_path_by_model_name(name=name)
                lr_scheduler = LearningRateScheduler(utils.scheduler)
                csv_logger = CSVLogger(model_training_history, append=True)
                model_check_point = ModelCheckpoint(path_to_saved_model,save_best_only=True)
                try:
                    if sample_weights is not None:
                        print("using sample weights " + "-#-"*10)
                        self.history = self.model.fit(X_train, Y_train, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=(X_test, Y_test), verbose=verbose_, callbacks=[csv_logger, model_check_point,lr_scheduler], sample_weight=sample_weights)
                    else:
                        self.history = self.model.fit(X_train, Y_train, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=(X_test, Y_test), verbose=verbose_, callbacks=[csv_logger, model_check_point,lr_scheduler])

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
            path_training_history = Path(os.path.join(get_patch_model_training_data_folder_path(),name+".csv"))
            path_to_saved_model = Path(os.path.join(get_abs_saved_models_folder_path(),name+".keras"))
            try:
                
                csv_logger = CSVLogger(path_training_history)
                # model_check_point = ModelCheckpoint(path_to_saved_model)
                lr_scheduler = LearningRateScheduler(utils.scheduler)
                model_check_point = ModelCheckpoint(path_to_saved_model,save_best_only=True)
                if sample_weights is not None:
                    print("using sample weights " + "-#-"*10)
                    self.history = self.model.fit(X_train, Y_train, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=(X_test, Y_test), verbose=verbose_, callbacks=[csv_logger, model_check_point], sample_weight=sample_weights)
                else:
                    self.history = self.model.fit(X_train, Y_train, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=(X_test, Y_test), verbose=verbose_, callbacks=[csv_logger, model_check_point, lr_scheduler])
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



    def predict_1_point(self, point, expected_patch):
        tensor_point = tf.convert_to_tensor([point])
        predicted_patch = np.argmax(self.model.predict(tensor_point))
        print(f"expected patch: {expected_patch}")
        print(f"predicted patch: {predicted_patch}")
        return [expected_patch, predicted_patch]

    def predict(self, points):
        predictions = self.model.predict(points)
        predicted_patches = [np.argmax(i) for i in predictions]
        return predicted_patches

    # @tf.function
    def Predict(self,points,n=96):
        # returns up to 'n' candidates per point.
        points=tf.convert_to_tensor(points)
        predictions = self.model.predict(points,verbose=0)
        predicted_patches = [sorted(enumerate(pred_i), key=lambda x: x[1], reverse=True)[:n] for pred_i in predictions]
        return np.array(predicted_patches)      # shape:(n_points,n,2). Las dimension gives the pair (patch,probability)

    def predict_and_give_secondary_estimate(self, points):
        predictions = self.model.predict(points)
        predicted_patches = []
        for i in predictions:
            top_prediction = np.argmax(i)
            # remove top prediction
            i[top_prediction] = 0
            second_top_prediction = np.argmax(i)
            result = [top_prediction, second_top_prediction]
            predicted_patches.append(result)
        return predicted_patches

    def predict_and_give_three_estimates(self, points):
        predictions = self.model.predict(points)
        # print("predictions: ", predictions)
        predicted_patches = []
        for i in predictions:
            top_prediction = np.argmax(i)
            # remove top prediction
            i[top_prediction] = 0
            second_top_prediction = np.argmax(i)
            i[second_top_prediction] = 0
            third_top_prediction = np.argmax(i)
            result = [top_prediction, second_top_prediction, third_top_prediction]
            predicted_patches.append(result)
        return predicted_patches
    def plot(self,func='loss',name='', save=False,add_to_title='' ,ext='jpg', show=True, loaded_model=False):

        model_training_history = get_patch_model_training_data_file_abs_path_by_model_name(name=self.name)
        self.settings["epoch"] = len(pd.read_csv(model_training_history).index)
        print(self.settings)
        if loaded_model:
            title = f"{func} over {len(self.history['epoch'].index)} epochs"
            data1 = self.history[func].tolist()
            data2 = self.history[f'val_{func}'].tolist()
            lowest_point1 = (len(data1)-1, data1[-1])
            lowest_point2 = (len(data2)-1, data2[-1])
            

        else:
            title = f"{func} over {self.settings['epoch']} epochs"
            data1 = self.history.history[func]
            data2 = self.history.history[f'val_{func}']
            lowest_point1 = (len(data1)-1, data1[-1])
            lowest_point2 = (len(data2)-1, data2[-1])


        if func != 'loss':
            title = f"Accuracy over {self.settings['epoch']} epochs"
        
        if add_to_title != '':
            title = title + "\n" + add_to_title

        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(8)
        fig.set_figwidth(15)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-2,top=2e+0)
        ax.set_yticks(np.arange(1e-2,2e+0, 0.1))
        plt.tick_params(axis='y', which='minor')
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))

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
        ax.grid(True,which="both") 
        ax.set_title(title)
        ax.set_xlabel("epochs")
        # ax.set_xticks(np.arange(0,len(data1)+1, len(data1)//10))
        ax.set_ylabel(func)
        
        if save:
                try:
                    if self.__create_file_name():
                        name = name + "-" + self.__create_file_name()
                except AttributeError:
                    pass
                
                plot_path = Path(os.path.join(get_abs_saved_plots_folder_path(),f"patch_model/{name}.{ext}"))
                # print("path to be saved: ", plot_path)
                plt.savefig(plot_path)

        if show:
            plt.show()



    def save_(self, name="patch_model"):

 

        path = Path(os.path.join(get_abs_saved_models_folder_path(),f"{name}.keras"))
        
        self.model.save(path)
    
    def save_training_and_validation_data(self, name="patch_model"):
        name = name + "-" + self.__create_file_name()
        path = Path(os.path.join(get_patch_model_training_data_folder_path(),f"{name}.csv"))
        df = pd.DataFrame(self.history.history)
        with open(path, mode='w') as f:
            df.to_csv(f)

    def print_settings(self):
        for key, value in self.settings.items():
            print(f"{key}:{value}")
    
    
    def __create_file_name(self):
        try:
            print("self.settings: ", self.settings)
            return f"shape-{self.settings['shape']}bs-{self.settings['batch_size']}"
        
        except AttributeError:
            return ""



    
class Experiment:

    def __init__(self, nl,NN, list_epochs, list_batch_sizes,list_optimizers, diamond=False, regularizer=False):
        self.nums_layers = nl
        self.list_num_neurons_per_layer = NN
        self.list_epochs = list_epochs
        self.list_batch_sizes = list_batch_sizes
        self.list_optimizers = list_optimizers
        self.regularizer = regularizer
        self.combinations = self.create_combinations_of_settings()

        if diamond:
            pass
        if regularizer:
            print("weights (kernel) regularizer L2 will be used for output layer")

    def create_combinations_of_settings(self):
        return list(itertools.product(self.nums_layers,
                                      self.list_num_neurons_per_layer,
                                      self.list_epochs,
                                      self.list_batch_sizes,
                                      self.list_optimizers))

    def run(self,data, save=False, name="patch-model-experiment"):
        X_train, X_test, Y_train, Y_test = data
        combinations_of_settings = self.create_combinations_of_settings()

        for setting_ in combinations_of_settings:

            # name_ = self.create_file_name_from_settings(setting_)
            print("current settings: ", setting_)
            number_of_layers, number_of_neurons, epochs_, batch_size_, optimizer_ = setting_
            NNShape = [number_of_neurons] * number_of_layers
            print("NNSHAPE = ", NNShape)
            patch_model = PatchClassificationModel(NNShape=NNShape, regularizer=self.regularizer)
            
            patch_model.compile(opt=optimizer_, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'])
            patch_model.train((X_train, X_test, Y_train, Y_test),epochs_, batch_size_, verbose_=1, name=name)

            
            if save:
                patch_model.plot(add_to_title="loss function: sparse_categorical_crossentropy" ,save=save,show=False, name=name)
                patch_model.save_(name=name)
                patch_model.save_training_and_validation_data(name=name)

    def __str__(self):
        return f"The Experiment will try all different combinations of the following:\n\
        Number of Layers:{self.nums_layers}\n\
        Number of neurons per layer:{self.list_num_neurons_per_layer}\n\
        Number of epcohs:{self.list_epochs}\n\
        Number of batches:{self.list_batch_sizes}\n\
        List of optimizers:{self.list_optimizers}\n"
        
# class PatchModelWeightsTrained

class PatchModelUtils:

    @classmethod
    def check_which_subsequent_guesses_are_correct(cls, path):



        X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=1, split=0.2, random_state=1)
        patch_model = tf.keras.models.load_model(path)

        results = { 
                    str(i):0 for i in range(97)
                    }
        # start_ = start
        # end_ = end
        wrong_predictions = 0
        number_of_samples = len(X_test)
        print("number of samples: ", number_of_samples)

        predictions = patch_model.predict(X_test)
        print("len of predictions: ", len(predictions))
        predictions_and_index_list = [utils.get_top_N_largest_nums_indices_in_list(i,N=96) for i in predictions]
        
        expectations = Y_test
        for i in range(len(predictions_and_index_list)):
            expected_patch = Y_test[i]
            position = 0
            for prediction in predictions_and_index_list[i].keys():
                if int(prediction) == int(expected_patch):
                    if position != 0:
                        wrong_predictions += 1
                    results[str(position)] += 1
                    continue
                position += 1
        # print("percentage of mistakes: %2.2f %% ()" %(wrong_predictions*100/number_of_samples))
        print(f"percentage of mistakes: {(wrong_predictions/number_of_samples):2.2%} ({wrong_predictions})")
        for key, value in results.items():
            print(f"percentage of guess number {key} being correct: {(value/wrong_predictions):2.2%} {(value)}")



    def train_a_model_with_output_layer_weights_affected_by_number_of_samples_per_patch(shape):
        patch_model = PatchClassificationModel(NNShape=shape)
        last_layer = patch_model.model.layers[-1]
        print("model.layers: ", patch_model.model.layers)


    @classmethod
    def add_L2_regularizer_to_output_layer_by_creating_a_new_model(cls, patch_model):
        shape = patch_model.settings["shape_as_list"]
        new_patch_model_with_regularizer = PatchClassificationModel(NNShape=shape, regularizer=True)
        old_model_weights = patch_model.model.get_weights()
        
        new_patch_model_with_regularizer.model.set_weights(old_model_weights)
        
        return new_patch_model_with_regularizer
    # def l1_reg(weight_matrix):
    #     return 0.01 * ops.sum(ops.absolute(weight_matrix))