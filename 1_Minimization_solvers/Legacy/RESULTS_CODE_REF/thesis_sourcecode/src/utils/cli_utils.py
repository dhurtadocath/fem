import os
import sys
import signal
import numpy as np
import tensorflow as tf
from keras.models import load_model
from .path_funcs import get_relative_saved_models_folder_path, get_abs_path, get_relative_data_folder_path

def get_list_of_available_models():
    saved_models = filter(lambda x: x!="__init__.py",os.listdir(get_abs_path(get_relative_saved_models_folder_path())))

    return saved_models

def parse_point_from_str(string):
    coords = string.split(",")
    return tf.convert_to_tensor([(float(coords[0]),float(coords[1]),float(coords[2]))])

def load_a_model(model_name):
    model_path = get_abs_path(os.path.join(get_relative_saved_models_folder_path(), model_name))
    model = load_model(model_path)
    return model

def test_patch_model_with_a_point(patch_model, point):
    if type(point) == str:
        point = parse_point_from_str(point)
    return np.argmax(patch_model.predict(point))


def load_and_test_a_patch_model(patch_model_name, point):
    parsed_point = parse_point_from_str(point)
    model = load_a_model(patch_model_name)
    return test_patch_model_with_a_point(patch_model=model, point=parsed_point)

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)