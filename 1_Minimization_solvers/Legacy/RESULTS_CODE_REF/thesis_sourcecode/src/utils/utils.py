import os
import re
import random
import time
import numpy as np
import pandas as pd
import collections
from pathlib import Path
from .path_funcs import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
from ..model_training import patch_model_settings 
from itertools import product

def isEven(n):
    return n % 2 == 0

def create_diamond_shape_using_powers_of_two(starting_num_neurons,n):
    result = []
    if isEven(n=n):
        first_half = [starting_num_neurons*2**i for i in range(n)]
        second_half = [starting_num_neurons*2**i for i in range(n, -1, -1)]
        result =  first_half + second_half
    else:
        first_half = [starting_num_neurons*2**i for i in range(n+1)]
        second_half = [starting_num_neurons*2**i for i in range(n-1, -1, -1)]
        result =  first_half + second_half
    return result

def print_training_results():
    path_to_patch_model_training_data = get_patch_model_training_data_folder_path()
    files = [i for i in os.listdir(path=path_to_patch_model_training_data) if "test" not in i and "rand" in i]
    print(files)
    print(len(files))
    for i in files:
        current_file_path = os.path.join(path_to_patch_model_training_data, i)
        df = pd.read_csv(current_file_path)
        print(i)
        print(df.shape)
        # print(df.columns)
        print(df.iloc[-1])
        # print(df.describe())
        print("-"*10)
        print( )

def get_indices_of_largest_N_numbers_in_a_list(list_):
    result = {}
    for i in range(len(list_)):
        result[i] = list_[i]
    result = sorted(result.items(),key=lambda x:x[1], reverse=True)
    return result


def get_top_N_largest_nums_indices_in_list(list_, N=5):
    result = {}
    count = 0
    for key, value in get_indices_of_largest_N_numbers_in_a_list(list_):
        result[key] = value
        count += 1
        if count == N:
            return result
            
    return result

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def moving_average2(input_vector, width):
    n = len(input_vector)
    result = np.zeros_like(input_vector)
    std = np.zeros_like(input_vector)
    for i in range(n):
        index_before = max(0, i-width)
        index_after = min(i+width, n)
        window = input_vector[index_before:index_after]
        result[i] = np.mean(window)
        
    return result

def plot_training_history(model_name, model_type,func='loss',plot_name_and_abs_path='',add_to_title='' ,
                          ext='jpg', show=True, y_range=(1e-4,2e+0), plot_smooth=False):
    if model_type == "p":
        path_to_csv = get_patch_model_training_data_file_abs_path_by_model_name(model_name)
    elif model_type == "sp":
        path_to_csv = get_surface_points_model_training_data_file_abs_path_by_model_name(model_name)
    elif model_type == "sd":
        path_to_csv = get_signed_distance_model_training_data_file_abs_path_by_model_name(model_name)
    df = pd.read_csv(path_to_csv)
    title = f"{func} over {len(df.index)} epochs"
    # print("loaded_model.history: ", loaded_model.history)
    # print("loaded_model.history.keys(): ", loaded_model.history.keys())
    # print("loaded_model.history.head(): ", loaded_model.history.head())
    data1 = df[func].tolist()
    data2 = df[f"val_{func}"].tolist()
    # print("data1: ", data1)
    lowest_point1 = (len(data1)-1, data1[-1])
    lowest_point2 = (len(data2)-1, data2[-1])


   
    if func != 'loss':
        title = f"Accuracy over {len(df.index)} epochs"
    
    if add_to_title != '':
        title = title + "\n" + add_to_title

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(8)
    fig.set_figwidth(15)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log")
    ax.set_ylim(bottom=y_range[0],top=y_range[1])
    plt.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))

    text1 = str(lowest_point1[1])
    text2 = str(lowest_point2[1])
    
    ax.plot(data1, linewidth=0.5, color='red', label=func+f'-(final = {text1})')
    
    ax.plot(data2, linewidth=0.5, color='blue', label=f'val_{func}'+f'-(final = {text2})')

    if plot_smooth:
        ax.plot(moving_average2(data1, 200), color='yellow', linewidth=3,label=func+f'-(final = {text1}) moving average')
    
        ax.plot(moving_average2(data2, 200), color='green',linewidth=3,  label=f'val_{func}'+f'-(final = {text2} moving average)')
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
    # ax.set_xticks(np.arange(0,100+1, 5))
    ax.set_ylabel(func)

    if plot_name_and_abs_path:
        print(f"plot_name_and_abs_path: {plot_name_and_abs_path}")
        plt.savefig(plot_name_and_abs_path)
    if show:
        plt.show()


def plot_mul_training_history(model_names, model_type,func='loss',plot_name_and_abs_path='',add_to_title='' ,
                          ext='jpg', show=True, y_range=(1e-4,2e+0), plot_smooth=False):
    model_paths = []
    if model_type == "p":
        for i in model_names:
            path_to_csv = get_patch_model_training_data_file_abs_path_by_model_name(i)
            model_paths.append(path_to_csv)

    elif model_type == "sp":
        for i in model_names:
            path_to_csv = get_surface_points_model_training_data_file_abs_path_by_model_name(i)
            model_paths.append(path_to_csv)
    elif model_type == "sd":
        for i in model_names:
            path_to_csv = get_signed_distance_model_training_data_file_abs_path_by_model_name(i)
            model_paths.append(path_to_csv)
    
    try:
        assert len(model_names) >= 2, "input list has less than 2 elements"
        fig, axs = plt.subplots((len(model_names)//2) + (len(model_names)%2),2)
        plots = [[i[0], i[1]] for i in product(range((len(model_names)//2) + (len(model_names)%2)), [0, 1])]
        if len(plots) == 2:
            plots = [0, 1]
        print(f"Plots: ", plots)
        for i in range(len(model_paths)): 
            df = pd.read_csv(model_paths[i])
            title = f"{func} over {len(df.index)} epochs"
            # print("loaded_model.history: ", loaded_model.history)
            # print("loaded_model.history.keys(): ", loaded_model.history.keys())
            # print("loaded_model.history.head(): ", loaded_model.history.head())
            data1 = df[func].tolist()
            data2 = df[f"val_{func}"].tolist()
            # print("data1: ", data1)
            lowest_point1 = (len(data1)-1, data1[-1])
            lowest_point2 = (len(data2)-1, data2[-1])


        
            if func != 'loss':
                title = f"Accuracy over {len(df.index)} epochs"
            
            if add_to_title != '':
                title = title + "\n" + add_to_title

            
            fig.set_figheight(10)
            fig.set_figwidth(15)
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            if len(plots) == 2:
                axs[plots[i]].set_yscale("log")
                axs[plots[i]].set_ylim(bottom=y_range[0],top=y_range[1])
                plt.tick_params(axis='y', which='minor')
                axs[plots[i]].yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
            else:
                axs[plots[i][0],plots[i][1]].set_yscale("log")
                axs[plots[i][0],plots[i][1]].set_ylim(bottom=y_range[0],top=y_range[1])
                plt.tick_params(axis='y', which='minor')
                axs[plots[i][0],plots[i][1]].yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
            text1 = str(lowest_point1[1])
            text2 = str(lowest_point2[1])
            
            # ax.plot(data1, linewidth=0.5, color='red', label=func+f'-(final = {text1})')
            
            if len(plots) == 2:

                if plot_smooth:
                    
                    #LOSS
                    axs[plots[i]].plot(data1, linewidth=0.5, color='salmon', label=f'{func}'+f'-(final = {text1}) + {str(i)[20:]}')
                    axs[plots[i]].plot(moving_average2(data1, 200), color='red',linewidth=3,  label=f'{func}'+f'-(moving average)')
                    
                    #VALIDATION LOSS
                    axs[plots[i]].plot(data2, linewidth=0.5, color='cornflowerblue', label=f'val_{func}'+f'-(final = {text2}) + {str(i)[20:]}')
                    axs[plots[i]].plot(moving_average2(data2, 200), color='darkblue',linewidth=3,  label=f'val_{func}'+f'-(moving average)')

                else:
                    #LOSS
                    axs[plots[i]].plot(data1, linewidth=0.5, color='red', label=f'{func}'+f'-(final = {text1}) + {str(i)[20:]}')
                    #VALIDATION LOSS
                    axs[plots[i]].plot(data2, linewidth=0.5, color='blue', label=f'val_{func}'+f'-(final = {text2}) + {str(i)[20:]}')

            else:
                
                if plot_smooth:
                    #LOSS
                    axs[plots[i][0],plots[i][1]].plot(data1, linewidth=0.5, color='salmon', label=f'{func}'+f'-(final = {text2}) + {str(i)[20:]}')
                    axs[plots[i][0],plots[i][1]].plot(moving_average2(data1, 200), color='red',linewidth=3,  label=f'{func}'+f'-(moving average)')
                    #VALIDATION LOSS
                    axs[plots[i][0],plots[i][1]].plot(data2, linewidth=0.5, color='cornflowerblue', label=f'val_{func}'+f'-(final = {text2}) + {str(i)[20:]}')
                    axs[plots[i][0],plots[i][1]].plot(moving_average2(data2, 200), color='darkblue',linewidth=3,  label=f'val_{func}'+f'-(moving average)')
                else:
                    #LOSS
                    axs[plots[i][0],plots[i][1]].plot(data1, linewidth=0.5, color='red', label=f'val_{func}'+f'-(final = {text2}) + {str(i)[20:]}')
                    #VALIDATION LOSS
                    axs[plots[i][0],plots[i][1]].plot(data2, linewidth=0.5, color='blue', label=f'val_{func}'+f'-(final = {text2}) + {str(i)[20:]}')

                # ax.plot(moving_average2(data1, 200), color='yellow', linewidth=3,label=func+f'-(final = {text1}) moving average')
            # final_text = AnchoredText(,loc)

            # ax.annotate(text1,
            #         xy=lowest_point1, xycoords='axes fraction',
            #         xytext=(-1.5,-1.5), textcoords='offset points',
            #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            # ax.annotate(text2,
            #         xy=lowest_point2, xycoords='axes fraction',
            #         xytext=(-1.5,1.5), textcoords='offset points',
            #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

            if len(plots) == 2:
                axs[plots[i]].legend()
                axs[plots[i]].grid(True,which="both") 
                axs[plots[i]].set_title(model_names[i]+"\n" +" "+ title)
                axs[plots[i]].set_xlabel("epochs")
                # axs[plots[i]].set_xticks(np.arange(0,100+1, 5))
                axs[plots[i]].set_ylabel(func)
            else:
                axs[plots[i][0],plots[i][1]].legend()
                axs[plots[i][0],plots[i][1]].grid(True,which="both") 
                axs[plots[i][0],plots[i][1]].set_title(model_names[i]+"\n"+" " + title)
                axs[plots[i][0],plots[i][1]].set_xlabel("epochs")
                # axs[plots[i][0],plots[i][1]].set_xticks(np.arange(0,100+1, 5))
                axs[plots[i][0],plots[i][1]].set_ylabel(func)

        if plot_name_and_abs_path:
            print(f"plot_name_and_abs_path: {plot_name_and_abs_path}")
            plt.savefig(plot_name_and_abs_path)
        if show:
            plt.show()

    except AssertionError:
        print("The input contains less than 2 models, this function plots 2 or more models' training histories!")
def get_batch_size_from_name(name):
    pattern = r"bs\-([0-9]*)"
    batch_size = re.search(pattern=pattern, string=name)
    return batch_size.group(1)

def get_shape_from_name(name):
    pattern = r"shape(\-([0-9]*))*"

    shape = re.search(pattern=pattern, string=str(name))
    shape = shape.group(0).replace("shape", "")
    shape = shape[1:]
    return shape

def get_shape_from_name_as_list(name):
    pattern = r"shape(\-([0-9]*))*"

    shape = re.search(pattern=pattern, string=str(name))
    shape = shape.group(0).replace("shape", "")
    shape = shape[1:]
    shape = [int(i) for i in shape.split("-") if i]
    return shape

def re_enumerate_epochs_in_csv_file(csv_file_path):
    df = pd.read_csv(csv_file_path, index_col=False)
    # print(f"df before:\n{df}")
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
    df = df.drop('epoch', axis=1)
    df = df.reset_index().rename(columns={'index': 'epoch'})
    # print(f"df after:\n{df}")
    df.to_csv(csv_file_path, index=False)


def bar_plot_patch_model_performance_for_all_patches(patch_model_name, data, plot_name=None):
    t1 = time.time()
    X_train, X_test, Y_train, Y_test = data
    # patch_model_abs_path = get_abs_saved_patch_models_folder_path_with_model_name(patch_model_name)
    patch_model = patch_model_settings.PatchClassificationModel(name=patch_model_name)
    predictions = patch_model.predict(X_test)
    patches = [i for i in range(96)]
    Y_train = Y_train.numpy().tolist()
    # print("Y_train = ", Y_train)
    # print("Y_train.numpy().tolist() = ", Y_train.numpy().tolist())
    training_samples_per_patch = dict(sorted(collections.Counter(Y_train).items()))
    print(f"training_samples_per_patch: ", training_samples_per_patch)
    correct_predictions = list(dict(sorted(collections.Counter([predictions[i] for i in range(len(predictions)) if Y_test[i] == predictions[i]]).items())).values())
    incorrect_predictions = list(dict(sorted(collections.Counter([predictions[i] for i in range(len(predictions)) if Y_test[i] != predictions[i]]).items())).values())
    print(f"correct_predictions: {correct_predictions}")
    print(f"incorrect_predictions: {incorrect_predictions}")
    predictions = {
        "correctly predicted": correct_predictions,
        "incorrectly predicted": incorrect_predictions
    }
    width=1.5
    fig, ax = plt.subplots(figsize=(20, 10))
    factor = 4
    ax.bar([factor*i for i in training_samples_per_patch.keys()], training_samples_per_patch.values(), color="orange", width=width, label="Training points")
    ax.bar([(factor*i + width) for i in patches], predictions["correctly predicted"], width=width, label="Correctly predicted", color="green")
    ax.bar([(factor*i + width) for i in patches], predictions["incorrectly predicted"], width=width, label="Incorrectly predicted", bottom=predictions["correctly predicted"], color="red")

    ax.set_title("Number of points used for training compared to numbers of correct and incorrect predictions on validation data")
    ax.set_xticks([factor*i for i in training_samples_per_patch.keys()])
    
    ax.set_xticklabels(([str(i) for i in patches]), rotation=90)
    ax.legend(loc="upper right")
    print(f"total time in seconds: {time.time() - t1}")
    print(f"name: {plot_name}")
    if plot_name is not None:
        plt.savefig(plot_name+".jpeg")
    # plt.show()


def bar_plot_patch_model_performance_for_all_patches_for_multiple_models(patch_models_names_list, data, plot_name=None):
    t1 = time.time()
    X_train, X_test, Y_train, Y_test = data
    Y_train = Y_train.numpy().tolist()
    Y_test = Y_test.numpy().tolist()
    training_samples_per_patch = dict(sorted(collections.Counter(Y_train).items()))
    training_samples_per_patch = collections.OrderedDict(sorted(training_samples_per_patch.items(), key=lambda kv: kv[1], reverse=True))

    validation_samples_per_patch = dict(sorted(collections.Counter(Y_test).items()))
    patches_and_percentage_of_incorrectly_predicted_val_points = {
        key: value for key, value in zip([i for i in range(96)], [0 for i in range(96)])
        }
    
    model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points = {
        key: value for key, value in zip(patch_models_names_list, [{},{},{}])
    }
    

    
    for patch_model_name in patch_models_names_list:
        # patch_model_abs_path = get_abs_saved_patch_models_folder_path_with_model_name(patch_model_name)
        patch_model = patch_model_settings.PatchClassificationModel(name=patch_model_name)
        predictions = patch_model.predict(X_test)
        incorrect_predictions = list(dict(sorted(collections.Counter([predictions[i] for i in range(len(predictions)) if Y_test[i] != predictions[i]]).items())).values())
        for i in range(96):
            val_len = validation_samples_per_patch[i]
            print("val_len: ", val_len)
            patches_and_percentage_of_incorrectly_predicted_val_points[i] = (incorrect_predictions[list(training_samples_per_patch.keys())[i]]/val_len) * 100
        model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points[patch_model_name] = patches_and_percentage_of_incorrectly_predicted_val_points
        patches_and_percentage_of_incorrectly_predicted_val_points = {
        key: value for key, value in zip([i for i in range(96)], [0 for i in range(96)])
        }
        
    width=1.5
    fig, ax = plt.subplots(figsize=(20, 10))
    factor = 4
    print(training_samples_per_patch.keys())
    ax.bar([factor*i for i in range(96)], training_samples_per_patch.values(), color="lightgrey", width=width, label="Training points")
    
    ax2 = ax.twinx()
    ax2.set_ylabel(r"% " + "of wrongly predicted validation points over all validation points")
    # ax2.set_ylim([0, 105])
    # ax2.set_yticks([i for i in range(0,101,5)])
    # ax2.set_yticklabels([f"{i}%" for i in range(0,101,5)])

    ax.set_xlabel("patches")
    ax.set_ylabel("Number of training points (bars)")
    ax.set_xticks([factor*i for i in range(96)])
    ax.set_xticklabels(([str(i) for i in training_samples_per_patch.keys()]), rotation=90)
    
    for model_name,patches_and_percentages  in model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points.items():
        print(f"for model {model_name}, patches and percentages: {patches_and_percentages}")
        if "sample_weights" in model_name:
            ax2.plot([factor*i for i in range(96)], [i for i in patches_and_percentages.values()], label="with sample weights", linewidth=3)
        elif "regularizer" in model_name:
            ax2.plot([factor*i for i in range(96)], [i for i in patches_and_percentages.values()], label="with L-2 regularizer on output layer", linewidth=3)
        
        else:
            ax2.plot([factor*i for i in range(96)], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3)
    
    plt.title("per-patch comparison of patch classification models with two hidden layesr of 512 neurons each")
    plt.legend()
    plt.grid(True)
    plt.show()

def scatter_plot_patch_model_performance_for_all_patches_for_multiple_models(patch_models_names_list, data, plot_name=None):
    t1 = time.time()
    X_train, X_test, Y_train, Y_test = data
    Y_train = Y_train.numpy().tolist()
    Y_test = Y_test.numpy().tolist()
    training_samples_per_patch = dict(sorted(collections.Counter(Y_train).items()))
    training_samples_per_patch = collections.OrderedDict(sorted(training_samples_per_patch.items(), key=lambda kv: kv[1], reverse=True))

    validation_samples_per_patch = dict(sorted(collections.Counter(Y_test).items()))
    patches_and_percentage_of_incorrectly_predicted_val_points = {
        key: value for key, value in zip([i for i in range(96)], [0 for i in range(96)])
        }
    
    model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points = {
        key: value for key, value in zip(patch_models_names_list, [{},{},{}])
    }
    

    
    for patch_model_name in patch_models_names_list:
        # patch_model_abs_path = get_abs_saved_patch_models_folder_path_with_model_name(patch_model_name)
        patch_model = patch_model_settings.PatchClassificationModel(name=patch_model_name)
        predictions = patch_model.predict(X_test)
        incorrect_predictions = list(dict(sorted(collections.Counter([predictions[i] for i in range(len(predictions)) if Y_test[i] != predictions[i]]).items())).values())
        for i in range(96):
            val_len = validation_samples_per_patch[i]
            # print("val_len: ", val_len)
            patches_and_percentage_of_incorrectly_predicted_val_points[i] = (incorrect_predictions[list(training_samples_per_patch.keys())[i]]/val_len) * 100
        model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points[patch_model_name] = patches_and_percentage_of_incorrectly_predicted_val_points
        patches_and_percentage_of_incorrectly_predicted_val_points = {
        key: value for key, value in zip([i for i in range(96)], [0 for i in range(96)])
        }
        
    width=1.5
    fig, ax = plt.subplots(figsize=(20, 10))
    factor = 4
    print(training_samples_per_patch.keys())
    # ax.bar([factor*i for i in range(96)], training_samples_per_patch.values(), color="lightgrey", width=width, label="Training points")
    
    # ax2 = ax.twiny()
    # ax2.set_xlabel("patches")
    # ax2.set_xticks([i for i in training_samples_per_patch.values()])
    # ax2.set_xticklabels(([str(i) for i in list(model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points.values())[0].keys()]), rotation=75)
    ax.set_ylim([0, 30])
    ax.set_yticks([i for i in range(0,31,5)])
    ax.set_yticklabels([f"{i}%" for i in range(0,31,5)])

    ax.set_xlabel("Number of training points per patch")
    ax.set_ylabel(r"% " + "of wrongly predicted validation points over all validation points")
    # ax.set_xticks([factor*i for i in range(96)])
    # minx = min(list(training_samples_per_patch.values()))
    # maxx = max(list(training_samples_per_patch.values()))
    # ax.set_xticks([i for i in range(minx, maxx)])
    # ax.set_xticklabels(([str(i) for i in training_samples_per_patch.keys()]), rotation=90)
    
    for model_name,patches_and_percentages  in model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points.items():
        # print(f"for model {model_name}, patches and percentages: {patches_and_percentages}")
        ax.scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3)
    
    plt.title("per-patch comparison of patch classification models with two hidden layesr of 512 neurons each")
    plt.legend()
    plt.grid(True)
    plt.show()

def scatter_plot_patch_model_performance_for_all_patches_for_multiple_models_subplots(patch_models_names_list, data, plot_name=None):
    t1 = time.time()
    X_train, X_test, Y_train, Y_test = data
    Y_train = Y_train.numpy().tolist()
    Y_test = Y_test.numpy().tolist()
    training_samples_per_patch = dict(sorted(collections.Counter(Y_train).items()))
    training_samples_per_patch = collections.OrderedDict(sorted(training_samples_per_patch.items(), key=lambda kv: kv[1], reverse=True))

    validation_samples_per_patch = dict(sorted(collections.Counter(Y_test).items()))
    patches_and_percentage_of_incorrectly_predicted_val_points = {
        key: value for key, value in zip([i for i in range(96)], [0 for i in range(96)])
        }
    
    model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points = {
        key: value for key, value in zip(patch_models_names_list, [{},{},{}])
    }
    

    
    for patch_model_name in patch_models_names_list:
        # patch_model_abs_path = get_abs_saved_patch_models_folder_path_with_model_name(patch_model_name)
        patch_model = patch_model_settings.PatchClassificationModel(name=patch_model_name)
        predictions = patch_model.predict(X_test)
        incorrect_predictions = list(dict(sorted(collections.Counter([predictions[i] for i in range(len(predictions)) if Y_test[i] != predictions[i]]).items())).values())
        for i in range(96):
            val_len = validation_samples_per_patch[i]
            # print("val_len: ", val_len)
            patches_and_percentage_of_incorrectly_predicted_val_points[i] = (incorrect_predictions[list(training_samples_per_patch.keys())[i]]/val_len) * 100
        model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points[patch_model_name] = patches_and_percentage_of_incorrectly_predicted_val_points
        patches_and_percentage_of_incorrectly_predicted_val_points = {
        key: value for key, value in zip([i for i in range(96)], [0 for i in range(96)])
        }
        
    width=1.5
    fig, ax = plt.subplots(2, 2)
    factor = 4
    print(training_samples_per_patch.keys())
    # ax.bar([factor*i for i in range(96)], training_samples_per_patch.values(), color="lightgrey", width=width, label="Training points")
    
    # ax2 = ax.twiny()
    # ax2.set_xlabel("patches")
    # ax2.set_xticks([i for i in training_samples_per_patch.values()])
    # ax2.set_xticklabels(([str(i) for i in list(model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points.values())[0].keys()]), rotation=75)


    # ax.set_xticks([factor*i for i in range(96)])
    # minx = min(list(training_samples_per_patch.values()))
    # maxx = max(list(training_samples_per_patch.values()))
    # ax.set_xticks([i for i in range(minx, maxx)])
    # ax.set_xticklabels(([str(i) for i in training_samples_per_patch.keys()]), rotation=90)
    models_plotted = []
    for model_name,patches_and_percentages  in model_names_and_percentages_of_each_patch_incorrectly_predicted_validation_points.items():
        # print(f"for model {model_name}, patches and percentages: {patches_and_percentages}")
        ax[0, 0].scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3)
        
        if model_name not in models_plotted and len(models_plotted) == 0:
            ax[0,1].scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3)
            ax[1,0].scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3, alpha=0.2)
            ax[1,1].scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3, alpha=0.2)
        if model_name not in models_plotted and len(models_plotted) == 1:
            ax[0,1].scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3, alpha=0.2)
            ax[1,0].scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3)
            ax[1,1].scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3, alpha=0.2)
        if model_name not in models_plotted and len(models_plotted) == 2:
            ax[0,1].scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3, alpha=0.2)
            ax[1,0].scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3, alpha=0.2)
            ax[1,1].scatter([training_samples_per_patch[i] for i in patches_and_percentages.keys()], [i for i in patches_and_percentages.values()], label=model_name, linewidth=3)

        models_plotted.append(model_name)
    
    for i in [[0,0],[0,1],[1,0],[1,1]]:

        ax[i[0],i[1]].set_ylim([0, 30])
        ax[i[0],i[1]].set_yticks([i for i in range(0,31,5)])
        ax[i[0],i[1]].set_yticklabels([f"{i}%" for i in range(0,31,5)])

        ax[i[0],i[1]].set_xlabel("Number of training points per patch")
        ax[i[0],i[1]].set_ylabel(r"% " + "of wrongly predicted validation points over all validation points")
        ax[i[0],i[1]].grid(True)
        ax[i[0],i[1]].legend()
    plt.suptitle("per-patch comparison of patch classification models with two hidden layesr of 512 neurons each")
    plt.show()


def print_avg_last_20_training_epochs_with_std(model_type="p"):
    
    abs_training_path = get_patch_model_training_data_folder_path()
    if model_type == "sp":
        abs_training_path = get_surface_points_model_training_data_folder_path()
    elif model_type == "sd":
        abs_training_path = get_signed_distance_model_training_data_folder_path()

    
    all_training_history_csv_files = os.listdir(abs_training_path)

    for file_name in all_training_history_csv_files:
        print("Current file: ", file_name)
        print( )
        full_file_path = Path(os.path.join(abs_training_path, file_name))
        file = pd.read_csv(full_file_path)
        val_loss_mean = file.loc[-20:, "val_loss"].mean()
        val_loss_std = file.loc[-20:, "val_loss"].std()
        trn_loss_mean = file.loc[-20:, "loss"].mean()
        trn_loss_std = file.loc[-20:, "loss"].std()
        print("val_loss mean: ","{:.6f}".format(val_loss_mean))
        print("val_loss std: ","{:.6f}".format(val_loss_std))
        print("trn_loss mean: ","{:.6f}".format(trn_loss_mean))
        print("trn_loss std: ","{:.6f}".format(trn_loss_std))
        print("table info val_loss: " + str("{:.6f}".format(val_loss_mean)) + " " + u'\u00B1' + " " + str("{:.6f}".format(val_loss_std)))
        print("table info trn_loss: " + str("{:.6f}".format(trn_loss_mean)) + " " + u'\u00B1' + " " + str("{:.6f}".format(trn_loss_std)))
        print("-"*10)
        print( )


def multi_model_voting_system(data, models_names):
    X_train, X_test, Y_train, Y_test = data


    models = [patch_model_settings.PatchClassificationModel(name=i) for i in models_names]
    models_and_predictions = {}
    final_prediction = []
    for i in models:
        models_and_predictions[i] = i.predict_and_give_three_estimates(X_test)
    correct_final_predictions = 0
    # print("True values: ", Y_test)
    counter = 0
    Y_test_as_list = Y_test.numpy().tolist()
    for val in range(len(Y_test)):
        print("-"*10)
        print(f"current point: {X_test[val]}")
        votes = {i:0 for i in range(96)}
        models_first_predictions = []
        for model, predictions in models_and_predictions.items():
            
            # if Y_test[val] != predictions[val][0]:
            for i in predictions[val]:
                votes[i] +=1
            models_first_predictions.append(predictions[val][0])
            # print("real value: ", Y_test_as_list[val])
            # print(f"{model.name} predictions: {predictions[val]}")
                    # print(f"{sec_model.name} predictions: {sec_predicts[val]}")

        vote_counts_of_firt_predictions = collections.Counter(models_first_predictions)
        max_first_prediction_vote = max(vote_counts_of_firt_predictions, key=vote_counts_of_firt_predictions.get)  
        min_first_prediction_vote = min(vote_counts_of_firt_predictions, key=vote_counts_of_firt_predictions.get)  
        if max(models_first_predictions) == min(models_first_predictions):
            final_vote = models_first_predictions[0]

        elif max_first_prediction_vote != min_first_prediction_vote:
            final_vote = max_first_prediction_vote
        else:
            final_vote = max(votes, key=votes.get)
        
        if final_vote == Y_test_as_list[val]:
            correct_final_predictions += 1
            final_prediction.append(final_vote)
                # print("final vote: ", final_prediction[-1])
                # print("-"*19)
        elif final_vote != Y_test_as_list[val]:
            print(votes)
            for model, predictions in models_and_predictions.items():
                print(f"{model.name} predictions: {predictions[val]}")

        print(f"correct answer: {Y_test_as_list[val]}, final vote: {final_vote}")
        
    print("accuracy: ", (correct_final_predictions/len(Y_test_as_list))*100)

def scheduler(epoch, lr):
    if (epoch > 0) and ((epoch%50) == 0):
        return lr * np.exp(-0.1)
    else:
        return lr

# def get_classification_error_per_patch(model_names, data):
#     X_train, X_test, Y_train, Y_test = data
#     Y_test_as_list = Y_test.numpy().tolist()
#     models = []
#     models_and_accuracy_per_patch = {}
#     for patch_model_name in model_names:
#         wrong_predictions_per_patch = {i:0 for i in range(96)}
#         model = patch_model_settings.PatchClassificationModel(name=patch_model_name)
#         models.append(model)
#         predictions = model.predict(X_test)
#         validation_samples_per_patch = collections.Counter(Y_test_as_list)
#         for i in range(len(Y_test_as_list)):
#             if Y_test_as_list[i] != predictions[i]:
#                 wrong_predictions_per_patch[Y_test_as_list[i]] += 1
#         models_and_accuracy_per_patch[patch_model_name] = wrong_predictions_per_patch
        

#         print(f"model: {patch_model_name}")
#         print("\n\n")
#         print(f"errors per patch: \n {models_and_accuracy_per_patch[patch_model_name]}")
    
#     models_and_their_best_patches = {}
#     for i in range(96):
#         model_names_and_current_best_patch = {}
#         # min_index = list(models_and_accuracy_per_patch.values())
        
#         for model_name, wrong_predictions_per_patch in models_and_accuracy_per_patch.items():
#             model_names_and_current_best_patch[model_name] = wrong_predictions_per_patch[i]
        
#         min_index = np.argmin(list(model_names_and_current_best_patch.values()))
#         best_model_for_this_patch = list(model_names_and_current_best_patch.keys())[min_index]
#         models_and_their_best_patches[best_model_for_this_patch] = i
    
#     print("models with their best patches:\n")
#     for model, best_patches in models_and_their_best_patches.items():
#         print(f"model: {model}")
#         print(f"best patches: {best_patches}")