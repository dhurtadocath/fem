import os
import keras
import time
import random
import argparse
import numpy as np
import pandas as pd
import keras.losses
import keras.optimizers
import tensorflow as tf
import utils.utils as utils
import moviepy.editor as mp
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from keras.utils import to_categorical
from model_training.patch_model_settings import Experiment
from model_training.patch_model_settings import PatchClassificationModel
from model_training.surface_points_model_settings import SurfacePointsModel
from data_processing.data_organized import get_training_and_testing_data_for_patch_model, collect_csv_files_into_one_df, split_df_based_on_patch,create_patch_model_training_data, plot_data, plot_bar_points_per_patch, get_N_random_points_per_patch_for_patch_model_training,get_training_and_testing_data_and_sample_weights_for_patch_model
from utils.path_funcs import get_abs_saved_models_folder_path, get_abs_path, get_abs_raw_data_folder_path, get_abs_path_of_package_root
# from model_training.patch_model_settings import Utils


# 10k points per patch = 960000 | full dataset = 8 960 000
# when amount == 0 takes full dataset
# X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=0.1, split=0.2, random_state=1) # gets training data for the whole dataset
# X_train, X_test, Y_train, Y_test = get_N_random_points_per_patch_for_patch_model_training(10000, random_state=1, split=0.2)
# print(get_abs_path_of_package_root())
# collect_csv_files_into_one_df()
# df = collect_csv_files_into_one_df()
# X_train, X_test, Y_train, Y_test = create_patch_model_training_data(df, amount=0.1, split=0.2, random_state=1)
# X_train_points,X_train_patches, X_test_points,X_test_patches, Y_train_p1, Y_train_p2, Y_test_p1,Y_test_p2 = get_training_and_testing_data(amount=0.5, split=0.2, model=1) # gets training data for the whole dataset
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)

# layers1 = [2]
# neurons1 = [512]#, 1024, 2048]
# opt = tf.keras.optimizers.Adam()
# # exp = Experiment(nl=layers1,NN=neurons1, list_epochs=[1], list_batch_sizes=[64],list_optimizers=[opt], regularizer=False)
# # print(exp.create_combinations_of_settings())
# # exp.run((X_train, X_test, Y_train, Y_test),save=True, name="patch_model_1_epochs-testttt")

# exp2 = Experiment(nl=layers1,NN=neurons1, list_epochs=[2000], list_batch_sizes=[64],list_optimizers=[opt], regularizer=True)
# # print(exp.create_combinations_of_settings())
# exp2.run((X_train, X_test, Y_train, Y_test),save=True, name="patch_model_2000_epochs_regularizer")

# neurons1 = [512, 1024, 2048, 4096]
# for i in neurons1:

#     name1 = f"patch_model_rand_sample_0.1-shape-{i}-{i}-bs-64"
#     name2 = f"patch_model_rand_sample_0.1_weights_regularizer-shape-{i}-{i}-bs-64"

#     model1 = PatchClassificationModel(name=name1)
#     model1.plot(name=name1, show=True, loaded_model=True, save=True)
#     model2 = PatchClassificationModel(name=name2)
#     model2.plot(name=name2, show=True, loaded_model=True, save=True)

# ###############
# NN_SHAPES = []
# starting_neurons = [96, 128]
# number_of_layers = [3, 4, 5, 6]
# for i in range(len(starting_neurons)):

#     nnshape = create_diamond_shape_using_powers_of_two(starting_num_neurons=starting_neurons[i],n=number_of_layers[i])
#     NN_SHAPES.append(nnshape)
#     # print("nnshape = ", nnshape)

# # for nn_shape in NN_SHAPES[1:]:
# nn_shape = create_diamond_shape_using_powers_of_two(starting_num_neurons=8, n=7)
# name="patch_model_rand_sample_0.1-"
# # # print("current nnshape training: ", nn_shape)
# patch_model = PatchClassificationModel(NNShape=[512, 512], regularizer=True)
# patch_model.compile(opt=opt,loss_="sparse_categorical_crossentropy",metrics_=['accuracy'])
# patch_model.train((X_train, X_test, Y_train, Y_test),epochs_=50, batch_size_=1024)
# patch_model.plot(show=True, save=False)
# patch_model.save_(name=name)
# patch_model.save_training_and_validation_data(name=name)
# data = collect_csv_files_into_one_df()
# data_per_patch = split_df_based_on_patch(data)
# total = 0
# for df in data_per_patch:
#     total += len(df.index)
#     print("for patch ", df["patch"].unique(), "there are  ", "{:,}###".format(len(df.index)), " total = ", total)
# print_training_results()

# data = collect_csv_files_into_one_df()
# X_train, X_test, Y_train, Y_test = get_N_random_points_per_patch_for_patch_model_training(10000, random_state=1, split=0.2)
# print("x_train.shape: ", X_train.shape)
# print("Y_train.shape: ", Y_train.shape)
# print("X_test.shape: ", X_test.shape)
# print("Y_test.shape: ", Y_test.shape)
# layers1 = [2]
# neurons1 = [512]#, 1024, 2048, 4096]
# opt = tf.keras.optimizers.Adam()
# exp = Experiment(nl=layers1,NN=neurons1, list_epochs=[5000], list_batch_sizes=[64],list_optimizers=[opt])
# print(exp.create_combinations_of_settings())
# exp.run((X_train, X_test, Y_train, Y_test),save=True, name="patch_model_rand_sample_20k_points_per_patch_5k_epochs")
# path = "C:/UniversityImportantFiles/Master/semester 4/thesis_sourcecode/src/model_training/saved_models/patch_model_rand_sample_10k_points_per_patch-shape-512-512-bs-64.keras"
# # # Utils.check_which_subsequent_guesses_are_correct(path=path)
# model = PatchClassificationModel.load_model(path=path)
# trn_predictions = model.predict(X_train)
# tst_predictions = model.predict(X_test)
# wrongly_predicted_training_points = [X_train[i] for i in range(len(X_train)) if np.argmax(trn_predictions[i]) != Y_train[i]]
# wrongly_predicted_testing_points = [X_train[i] for i in range(len(X_test)) if np.argmax(tst_predictions[i]) != Y_test[i]]
# plot_data(training_data=wrongly_predicted_training_points, testing_data=wrongly_predicted_testing_points)
# list_of_wrong_guesses_trn = [i for i in X_train]

def train_new_patch_model_raw(shape=None, name=None, epochs=100, verbose=None): #default: shape=None, name=None, epochs=100
    if verbose is not None:
        print(f"Now training a raw new patch model.")
        print(f"Model shape: {shape}, Epochs: {epochs}, Name: {name}")
    


    X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=0.1, split=0.2, random_state=1) # gets training data for the whole dataset
    opt = tf.keras.optimizers.Adam()
    patch_model_test = PatchClassificationModel(NNShape=shape)
    patch_model_test.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=False)
    patch_model_test.train((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=1, name=name)


def retrain_existing_patch_model_raw(name=None, epochs=100, verbose=None):
    if verbose is not None:
        print(f"Now training an existing patch model.")
        print(f"Epochs: {epochs}, name: {name}")
    X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=0.1, split=0.2, random_state=1) # gets training data for the whole dataset
    opt = tf.keras.optimizers.Adam()
    patch_model_test = PatchClassificationModel(name=name)
    patch_model_test.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=False)
    patch_model_test.retrain_existing_model((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=1, name=name)


def train_new_patch_model_with_sample_weights(shape=None, name=None, epochs=100,verbose=None):
    if verbose is not None:
        print(f"Now training a new patch model with sample_weights.")
        print(f"Model shape: {shape}, Epochs: {epochs}, Name: {name}")
    X_train, X_test, Y_train, Y_test, sample_weights_for_training_data = get_training_and_testing_data_and_sample_weights_for_patch_model(amount=0.1, split=0.2, random_state=1)
    opt = tf.keras.optimizers.Adam()
    patch_model_test = PatchClassificationModel(NNShape=shape)
    patch_model_test.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=True)
    patch_model_test.train((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=1, name=name, sample_weights=sample_weights_for_training_data)


def retrain_existing_patch_model_with_sample_weights(name=None, epochs=100, verbose=None):
    if verbose is not None:
        print(f"Now training an existing patch model with sample_weights.")
        print(f"epochs: {epochs}, name: {name}")
    X_train, X_test, Y_train, Y_test, sample_weights_for_training_data = get_training_and_testing_data_and_sample_weights_for_patch_model(amount=0.1, split=0.2, random_state=1)
    opt = tf.keras.optimizers.Adam()
    patch_model_test = PatchClassificationModel(name=name)
    patch_model_test.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=True)
    patch_model_test.retrain_existing_model((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=1, name=name, sample_weights=sample_weights_for_training_data)


def train(args):
    if args.model_type == "p":
        shape = args.shape
        epochs = args.epochs
        name = args.name
        verbose = args.verbose

        if args.weights:
            train_new_patch_model_with_sample_weights(shape=shape, name=name, epochs=epochs, verbose=verbose)
        else:
            train_new_patch_model_raw(shape=shape, name=name, epochs=epochs, verbose=verbose)
        
    elif args.model_type == "sp":
        pass

def retrain(args):
    if args.model_type == "p":
        epochs = args.epochs
        name = args.name
        verbose = args.verbose
        print(args)
        if args.weights:
            retrain_existing_patch_model_with_sample_weights(name=name, epochs=epochs, verbose=verbose)
        else:
            retrain_existing_patch_model_raw(name=name, epochs=epochs, verbose=verbose)
        
    elif args.model_type == "sp":
        pass

def plot_model_perfomance_for_all_patches(patch_model_name, plot_name):
    X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=0.1, split=0.2, random_state=1) # gets training data for the whole dataset
    utils.bar_plot_patch_model_performance_for_all_patches(patch_model_name=patch_model_name, data=(X_train, X_test, Y_train, Y_test), plot_name=plot_name)


def plot_ppp(args):
    print(args)
    plot_model_perfomance_for_all_patches(patch_model_name=args.model_name, plot_name=args.plot_name)

def main():
    general_parser = argparse.ArgumentParser(description='thesis_sourcecode program.')
    subparsers = general_parser.add_subparsers(required=True)

    # create the parser for the "train" command
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-s','--shape', type=int,nargs='+', help='<Required> Set Number of hidden layers and their nodes', required=True)
    train_parser.add_argument('-mt','--model-type', type=str, help='<Required> Set type of model to train', required=True)
    train_parser.add_argument('-w','--weights', help='<Optional> Set whether to train model with sample_weights', action='store_true')
    train_parser.add_argument('-e','--epochs', type=int,help='<Required> Set number of epochs to train the new model', required=True)
    train_parser.add_argument('-n','--name', type=str,help='<Required> Set name-prefix for the training history and model files', required=True)
    train_parser.add_argument('-v', '--verbose',action='store_true')
    train_parser.set_defaults(func=train)

    # create the parser for the "retrain" command
    retrain_parser = subparsers.add_parser('retrain')
    # train_parser.add_argument('-s','--shape', type=int,nargs='+', help='<Required> Set Number of hidden layers and their nodes', required=True)
    retrain_parser.add_argument('-mt','--model-type', type=str, help='<Required> Set type of model to train', required=True)
    retrain_parser.add_argument('-w','--weights', help='<Optional> Set whether to train model with sample_weights', action='store_true')
    retrain_parser.add_argument('-e','--epochs', type=int,help='<Required> Set number of epochs to train the new model', required=True)
    retrain_parser.add_argument('-n','--name', type=str,help='<Required> Set name-prefix for the training history and model files', required=True)
    retrain_parser.add_argument('-v', '--verbose',action='store_true')
    retrain_parser.set_defaults(func=retrain)


    plot_ppp_parser = subparsers.add_parser('plot_ppp')
    plot_ppp_parser.add_argument('-mn','--model-name', type=str,help='<Required> Set name-prefix for the training history and model files', required=True)
    plot_ppp_parser.add_argument('-pn','--plot-name', type=str,help='<Required> Set name-prefix for the plot name and save it')
    plot_ppp_parser.set_defaults(func=plot_ppp)


    parse_args_output = general_parser.parse_args()
    parse_args_output.func(parse_args_output)



        
if __name__ == "__main__":
   
    main()
    # name = "testing-model-check-point-callback"
    # full_model_to_load_path = get_abs_saved_patch_models_folder_path_with_model_name(name=name)       
    # print("full_model_to_load_path: ", full_model_to_load_path)

    # plot_training_history("patch_model_2000_epochs_regularizer-shape-512-512-bs-64", show=True)
