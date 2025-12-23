import os
import sys
import keras
import keras.backend as K
import time
import random
import argparse
import numpy as np
import pandas as pd
import keras.losses
import keras.optimizers
import tensorflow as tf
import moviepy.editor as mp
import matplotlib.pyplot as plt
import src.utils.utils as utils
from matplotlib.animation import PillowWriter
from keras.utils import to_categorical
from src.model_training.patch_model_settings import Experiment
from src.model_training.patch_model_settings import PatchClassificationModel
from src.model_training.patch_model_settings import PatchModelUtils
from src.model_training.surface_points_model_settings import SurfacePointsModel, SurfacePointsModelForOnePatch
from src.model_training.signed_distance_model_settings import SignedDistanceModel
from src.data_processing.data_organized import get_training_and_testing_data_for_patch_model, collect_csv_files_into_one_df,\
      split_df_based_on_patch,create_patch_model_training_data, plot_data, plot_bar_points_per_patch,\
          get_N_random_points_per_patch_for_patch_model_training,get_training_and_testing_data_and_sample_weights_for_patch_model,\
          get_training_and_testing_data_for_surface_point_model_for_one_patch, get_training_and_testing_data_for_signed_distance_model
from src.utils.path_funcs import get_abs_saved_models_folder_path, get_abs_path, get_abs_raw_data_folder_path, get_abs_path_of_package_root
# from model_training.patch_model_settings import Utils
from tests.unit_tests.add_L2_regularizer_to_output_layer_of_patch_classification_model_test import *
from src.utils.decorators import cprofile_function
# from ...src.model_training.patch_model_settings import PatchModelUtils

import cProfile
import pstats
import io
from contextlib import redirect_stdout


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


def train_new_patch_model_raw(shape=None, name=None, epochs=100, verbose=None, regularizer=False, data_set_amount=0.1, edges_only=False): #default: shape=None, name=None, epochs=100
    
    X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=data_set_amount, split=0.2, random_state=1,edges_only=edges_only) # gets training data for the whole dataset
    if verbose is not None:
        print(f"Now training a raw new patch model.")
        print(f"Model shape: {shape}, Epochs: {epochs}, Name: {name}")
        print(f"X_train.shape: ", X_train.shape)
        print(f"X_test.shape: ", X_test.shape)
        print(f"Y_train.shape: ", Y_train.shape)
        print(f"Y_test.shape: ", Y_test.shape)


    opt = tf.keras.optimizers.Adam()
    patch_model = PatchClassificationModel(NNShape=shape,regularizer=regularizer)
    patch_model.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=False)
    patch_model.train((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=verbose, name=name)

def train_new_surface_points_model_raw(shape=None, name=None, epochs=100, patch=25,verbose=None, regularizer=False, data_set_amount=0.1): #default: shape=None, name=None, epochs=100
    X_train_points, Y_train_sp, X_test_points, Y_test_sp = get_training_and_testing_data_for_surface_point_model_for_one_patch(patch=patch,amount=data_set_amount, split=0.2, random_state=1)
    if verbose is not None:
        print(f"Now training a raw new surface points model.")
        print(f"Model shape: {shape}, Epochs: {epochs}, Name: {name}")
        print(f"X_train_points.shape: ", X_train_points.shape)
        print(f"X_test_points.shape: ", X_test_points.shape)
        print(f"Y_train_sp.shape: ", Y_train_sp.shape)
        print(f"Y_test_sp.shape: ", Y_test_sp.shape)

    opt = tf.keras.optimizers.Adam()
    sp_model = SurfacePointsModelForOnePatch(NNShape=shape,regularizer=regularizer)
    sp_model.compile(opt=opt, loss_="mae", metrics_=["mae"])
    sp_model.train((X_train_points, X_test_points,  Y_train_sp, Y_test_sp), epochs, batch_size_=64, verbose_=verbose, name=name)

def retrain_existing_sp_model_raw(name=None, epochs=100, patch=25,verbose=None,data_set_amount=0.1):
    X_train_points, Y_train_sp, X_test_points, Y_test_sp = get_training_and_testing_data_for_surface_point_model_for_one_patch(patch=patch,amount=data_set_amount, split=0.2, random_state=1)
    if verbose is not None:
        print(f"Now training an existing surface points model on patch {patch}.")
        print(f"Epochs: {epochs}, name: {name}")
        print(f"X_train_points.shape: ", X_train_points.shape)
        print(f"X_test_points.shape: ", X_test_points.shape)
        print(f"Y_train_sp.shape: ", Y_train_sp.shape)
        print(f"Y_test_sp.shape: ", Y_test_sp.shape)

    opt = tf.keras.optimizers.Adam()
    sp_model = SurfacePointsModelForOnePatch(name=name)
    sp_model.compile(opt=opt, loss_="mae", metrics_=["mae"])
    sp_model.retrain_existing_model((X_train_points, X_test_points,  Y_train_sp, Y_test_sp), epochs, batch_size_=64,verbose_=verbose, name=name)



def retrain_existing_patch_model_raw(name=None, epochs=100, verbose=None,data_set_amount=0.1, edges_only=False):
    X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=data_set_amount, split=0.2, random_state=1, edges_only=edges_only) # gets training data for the whole dataset
    if verbose is not None:
        print(f"Now training an existing patch model.")
        print(f"Epochs: {epochs}, name: {name}")
        print(f"X_train.shape: ", X_train.shape)
        print(f"X_test.shape: ", X_test.shape)
        print(f"Y_train.shape: ", Y_train.shape)
        print(f"Y_test.shape: ", Y_test.shape)

    opt = tf.keras.optimizers.Adam()
    patch_model = PatchClassificationModel(name=name)
    patch_model.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=False)
    patch_model.retrain_existing_model((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=verbose, name=name)


def train_new_patch_model_with_sample_weights(shape=None, name=None, epochs=100,verbose=None, regularizer=False, data_set_amount=0.1):
    X_train, X_test, Y_train, Y_test, sample_weights_for_training_data = get_training_and_testing_data_and_sample_weights_for_patch_model(amount=data_set_amount, split=0.2, random_state=1)
    if verbose is not None:
        print(f"Now training a new patch model with sample_weights.")
        print(f"Model shape: {shape}, Epochs: {epochs}, Name: {name}")
        print(f"X_train.shape: ", X_train.shape)
        print(f"X_test.shape: ", X_test.shape)
        print(f"Y_train.shape: ", Y_train.shape)
        print(f"Y_test.shape: ", Y_test.shape)

    opt = tf.keras.optimizers.Adam()
    patch_model = PatchClassificationModel(NNShape=shape, regularizer=regularizer)
    patch_model.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=True)
    patch_model.train((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=verbose, name=name, sample_weights=sample_weights_for_training_data)


def retrain_existing_patch_model_with_sample_weights(name=None, epochs=100, verbose=None, data_set_amount=0.1):
    X_train, X_test, Y_train, Y_test, sample_weights_for_training_data = get_training_and_testing_data_and_sample_weights_for_patch_model(amount=data_set_amount, split=0.2, random_state=1)
    if verbose is not None:
        print(f"Now training an existing patch model with sample_weights.")
        print(f"epochs: {epochs}, name: {name}")
        print(f"X_train.shape: ", X_train.shape)
        print(f"X_test.shape: ", X_test.shape)
        print(f"Y_train.shape: ", Y_train.shape)
        print(f"Y_test.shape: ", Y_test.shape)
    opt = tf.keras.optimizers.Adam()
    patch_model = PatchClassificationModel(name=name)
    patch_model.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=True)
    patch_model.retrain_existing_model((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=verbose, name=name, sample_weights=sample_weights_for_training_data)


def train_new_signed_distance_model_raw(shape=None, name=None, epochs=100,verbose=None, regularizer=False, data_set_amount=0.1):
    X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_signed_distance_model(amount=data_set_amount, split=0.2, random_state=1)
    if verbose is not None:
        print(f"Now training a new signed distance model.")
        print(f"Model shape: {shape}, Epochs: {epochs}, Name: {name}")
        print(f"X_train.shape: ", X_train.shape)
        print(f"X_test.shape: ", X_test.shape)
        print(f"Y_train.shape: ", Y_train.shape)
        print(f"Y_test.shape: ", Y_test.shape)
    opt = tf.keras.optimizers.Adam()
    patch_model = SignedDistanceModel(NNShape=shape, regularizer=regularizer)
    patch_model.compile(opt=opt, loss_="mae", metrics_=['mae'])
    patch_model.train((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=verbose, name=name)

def retrain_existing_signed_distance_model_raw(shape=None, name=None, epochs=100,verbose=None, regularizer=False, data_set_amount=0.1):
    X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_signed_distance_model(amount=data_set_amount, split=0.2, random_state=1)
    
    if verbose is not None:
        print(f"Now training an existing signed distance model.")
        print(f"Model shape: {shape}, Epochs: {epochs}, Name: {name}")
        print(f"X_train.shape: ", X_train.shape)
        print(f"X_test.shape: ", X_test.shape)
        print(f"Y_train.shape: ", Y_train.shape)
        print(f"Y_test.shape: ", Y_test.shape)
    opt = tf.keras.optimizers.Adam()
    patch_model = SignedDistanceModel(NNShape=shape, regularizer=regularizer)
    patch_model.compile(opt=opt, loss_="mae", metrics_=['mae'])
    patch_model.retrain_existing_model((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=verbose, name=name)

@cprofile_function("testingProfiler2_as_decorator")
def train(args):
    shape = args.shape
    epochs = args.epochs
    name = args.name
    regularizer = args.regularizer
    verbose = args.verbose
    data_set_amount = args.data_set
    patch = args.sp_patch
    edges_only = args.edges_only

    if args.model_type == "p":
        if args.weights:
            train_new_patch_model_with_sample_weights(shape=shape, name=name, epochs=epochs, verbose=verbose, regularizer=regularizer,data_set_amount=data_set_amount)
        else:
            train_new_patch_model_raw(shape=shape, name=name, epochs=epochs, verbose=verbose, regularizer=regularizer,data_set_amount=data_set_amount, edges_only=edges_only)
        
    elif args.model_type == "sp":
        assert patch is not None, "Training a surface points model requires a patch to be chosen for training."
        if args.weights:
            pass
        else:
            train_new_surface_points_model_raw(shape=shape, name=name, epochs=epochs,patch=patch, verbose=verbose, regularizer=regularizer,data_set_amount=data_set_amount)

    elif args.model_type == "sd":
        train_new_signed_distance_model_raw(shape=shape, name=name, epochs=epochs, verbose=verbose, regularizer=regularizer,data_set_amount=data_set_amount)
        
def retrain(args):
    epochs = args.epochs
    name = args.name
    verbose = args.verbose
    data_set_amount = args.data_set
    sp_patch = args.sp_patch
    edges_only = args.edges_only
    print(f"name: {name}")
    print(f"type(name): {type(name)}")
    if args.model_type == "p":

        if args.weights:
            retrain_existing_patch_model_with_sample_weights(name=name, epochs=epochs, verbose=verbose,data_set_amount=data_set_amount)
        else:
            retrain_existing_patch_model_raw(name=name, epochs=epochs, verbose=verbose,data_set_amount=data_set_amount, edges_only=edges_only)
        
    elif args.model_type == "sp":
        retrain_existing_sp_model_raw(name=name,epochs=epochs,patch=sp_patch,verbose=verbose,data_set_amount=data_set_amount)

    elif args.model_type == "sd":
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
    train_parser.add_argument('-p','--sp-patch', type=int, help='<Optional> Set patch on which surface points model will be trained (REQUIRED only for surface points model)', required=False)
    train_parser.add_argument('-w','--weights', help='<Optional> Set whether to train model with sample_weights', action='store_true')
    train_parser.add_argument('-ds','--data-set',type=float, help='<Optional> Set how much of the dataset to use for training. 0 means use full dataset. default is 0.1')
    train_parser.add_argument('-r','--regularizer', help='<Optional> Set whether to train model with an L2 regularizer on the output layer', action='store_true')
    train_parser.add_argument('-e','--epochs', type=int,help='<Required> Set number of epochs to train the new model', required=True)
    train_parser.add_argument('-n','--name', type=str,help='<Required> Set name-prefix for the training history and model files', required=True)
    train_parser.add_argument('-v', '--verbose',action='store_true')
    train_parser.add_argument('-E','--edges-only',action='store_true')
    train_parser.set_defaults(func=train)

    # create the parser for the "retrain" command
    retrain_parser = subparsers.add_parser('retrain')
    # train_parser.add_argument('-s','--shape', type=int,nargs='+', help='<Required> Set Number of hidden layers and their nodes', required=True)
    retrain_parser.add_argument('-mt','--model-type', type=str, help='<Required> Set type of model to train', required=True)
    retrain_parser.add_argument('-p','--sp-patch', type=int, help='<Optional> Set patch on which surface points model will be trained (REQUIRED only for surface points model)', required=False)
    retrain_parser.add_argument('-w','--weights', help='<Optional> Set whether to train model with sample_weights', action='store_true')
    retrain_parser.add_argument('-ds','--data-set',type=float, help='<Optional> Set how much of the dataset to use for training. 0 means use full dataset. default is 0.1')
    retrain_parser.add_argument('-e','--epochs', type=int,help='<Required> Set number of epochs to train the new model', required=True)
    retrain_parser.add_argument('-n','--name', type=str,help='<Required> Set name-prefix for the training history and model files', required=True)
    retrain_parser.add_argument('-v', '--verbose',action='store_true')
    retrain_parser.add_argument('-E','--edges-only',action='store_true')
    retrain_parser.set_defaults(func=retrain)


    plot_ppp_parser = subparsers.add_parser('plot_ppp')
    plot_ppp_parser.add_argument('-mn','--model-name', type=str,help='<Required> Set name-prefix for the training history and model files', required=True)
    plot_ppp_parser.add_argument('-pn','--plot-name', type=str,help='<Required> Set name-prefix for the plot name and save it')
    plot_ppp_parser.set_defaults(func=plot_ppp)


    parse_args_output = general_parser.parse_args()
    parse_args_output.func(parse_args_output)



@cprofile_function("testingProfiler2_as_decorator")
def profiling():
    print("Hello world!")
    print(2**9)

if __name__ == "__main__":
    # utils.print_avg_last_20_training_epochs_with_std(model_type="sp")
    main()
    # utils.plot_training_history("patch_model_training_on_full_data_first_then_training_on_edges_with_full_data_validation","p")
    # X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_signed_distance_model(amount=0.1, split=0.2, random_state=1) # gets training data for the whole dataset

    # model = SignedDistanceModel(name="testing_second_sd_model-shape-512-512-bs-64")
    # random_indices = np.random.randint(50000, size=10)
    # print(f"indices: {random_indices}")
    # points = tf.convert_to_tensor([X_test[i] for i in random_indices])
    # print("points: ", points)
    # predictions = model.predict(points)
    # print(f"predictions: ")
    # tf.print(predictions, output_stream=sys.stderr)
    # print(f"Actual values: {[Y_test[i] for i in random_indices]}")
    # print(f"Actual values: \n{Y_test[:10]}")

    # utils.plot_training_history(model_name="testing_first_sd_model-shape-512-512-bs-64",model_type="sd", plot_smooth=True)
    # utils.plot_mul_training_history(model_names= [
    #     "surface_points_model_patch_25-shape-2048-2048-2048-2048-bs-64-epochs-1000",
    #     "surface_points_model_patch_25-shape-512-512-bs-64-epochs-1000",
    #     "surface_points_model_rand_sample_0.1_epochs_2000_from_start_to_finish-shape-512-512-bs-64",
    # ], model_type="sp", plot_smooth=True,y_range=(1e-3,1e-1))
    # utils.plot_training_history(model_name="surface_points_model_patch_25-shape-512-512-bs-64",model_type="sp", plot_smooth=True)
    # model_names = [
    #                "patch_model_rand_sample_10k_points_per_patch-shape-512-512-bs-64",
    #                "patch_model_rand_sample_0.1-shape-1024-1024-bs-64",
    #                "patch_model_rand_sample_0.1-shape-2048-2048-bs-64",
    #                "patch_model_rand_sample_0.1-shape-4096-4096-bs-64",
    #                "patch_model_rand_sample_0.1--shape-16384-bs-64",
    #                "patch_model_rand_sample_0.1--shape-8192-bs-64",
    #                ]

    # utils.multi_model_voting_system(data=(X_train, X_test, Y_train, Y_test), models_names=model_names)
    # point_to_test = (-1.4664625, -1.4621778, 1.7856798)
    # correct answer: 95, final vote: 94
    # models to test:
    # patch_model_rand_sample_0.1-shape-2048-2048-bs-64
    # patch_model_rand_sample_10k_points_per_patch-shape-1024-1024-bs-64
    # patch_model_rand_sample_0.1-shape-1024-1024-bs-64
    # patch_model_rand_sample_10k_points_per_patch-shape-2048-2048-bs-64
    # profiling()
    # model_name = "patch_model_rand_sample_0.1-shape-2048-2048-bs-64"
    # model = PatchClassificationModel(name=model_name)
    # print(model.predict_and_give_three_estimates([point_to_test]))
    # print(model.model.predict(point_to_test))

    # # opt = tf.keras.optimizers.Adam()
    # ###############
    # ds = 0.1
    # X_train1, X_test1, Y_train1, Y_test1 = get_training_and_testing_data_for_patch_model(amount=0, split=0.2, random_state=1, edges_only=True) # gets training data for the whole dataset
    # X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=0.1, split=0.2, random_state=1) # gets training data for the whole dataset
    

    # X_test_total = tf.concat([X_test, X_test1],0)
    # Y_test_total = tf.concat([Y_test, Y_test1],0)

    # name="patch_model_testing_load_reload_keras"
    # model = PatchClassificationModel([512, 512])
    # learning_rate = 0.001
    # model.compile(opt=keras.optimizers.Adam(),loss_="sparse_categorical_crossentropy",metrics_=['accuracy'],sample_weight=False)
    # model.train(data=(X_train, X_test, Y_train, Y_test), epochs_=2, batch_size_=64, verbose_=1, name=name)
    # reloaded_model = PatchClassificationModel(name=name)
    # model to plot: "patch_model_training_on_full_data_first_then_training_on_edges_with_full_data_validation"
    # reloaded_model.retrain_existing_model(data=(X_train, X_test,Y_train,Y_test),  epochs_=2, batch_size_=64, verbose_=1,name=name)

    # model.train(data=(X_train, X_test, Y_train, Y_test),epochs_=100, batch_size_=32,name=f"patch_classification_rand_sample_{ds}_sigmoid_glorot_init_lr_{learning_rate}")
    # model_names = ["patch_model_rand_sample_0.1--shape-512-512-bs-64-200-epochs",
    #                "patch_model_rand_sample_0.1_weights_regularizer-shape-512-512-bs-64",
    #                "patch_model_rand_sample_0.1_sample_weights-shape-512-512-bs-64",
    #                "patch_model_rand_sample_0.1-shape-512-512-bs-64",
    #                "patch_model_leaky_relu_alpha_0.01-shape-512-512-bs-64",
    #                "patch_model_sigmoid-shape-512-512-bs-64"]
    # utils.get_classification_error_per_patch(model_names=model_names,data=(X_train, X_test, Y_train, Y_test))
    # utils.scatter_plot_patch_model_performance_for_all_patches_for_multiple_models_subplots(patch_models_names_list=model_names,
    #             data=(X_train, X_test, Y_train, Y_test))
    ###############
    # model_name = "surface_points_model_test_patch_25"
    # # model_name = "patch_model_2000_epochs-"
    # # utils.plot_training_history(model_name=model_name,model_type="sp", plot_smooth=False)
    # # loaded_patch_model = PatchClassificationModel(name=model_name)
    # loaded_patch_model = SurfacePointsModelForOnePatch(name=model_name)
    # X_train_points, Y_train_sp, X_test_points, Y_test_sp = get_training_and_testing_data_for_surface_point_model_for_one_patch(patch=25,amount=0, split=0.2, random_state=1)
    # print(loaded_patch_model.predict(X_test_points))
    # print(Y_test_sp)

    # utils.plot_training_history(model_name=model_name, model_type="sp")
    # X_train_points, Y_train_sp, X_test_points, Y_test_sp = get_training_and_testing_data_for_surface_point_model_for_one_patch(amount=0,patch=11, split=0.2, random_state=1) # gets training data for the whole dataset

    # test_sp_model_name = "surface_points_model_patch_11-shape-512-512-bs-64"
    # # test_sp_model_name = "surface_points_model_patch_0_testing-shape-512-512-bs-64"
    # sp_model = SurfacePointsModelForOnePatch(name=test_sp_model_name)
    # print("X_test:", X_test_points)
    # print("Y_test:", Y_test_sp)
    # regressed_vals = sp_model.predict(X_test_points)
    # for i in range(5):
    #     print(f"Y_test[{i}]=    {Y_test_sp[i]}")
    #     print(f"predictions =", regressed_vals[i])
    
    pass