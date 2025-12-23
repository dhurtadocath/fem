import os
from pathlib import Path
from .decorators import make_output_path_obj

# from importlib_resources import path 
@make_output_path_obj
def get_abs_path_of_package_root():
    current_dir = os.path.dirname(os.path.realpath(__file__)) # get current abs path to this file
    abs_path = os.path.split(os.path.split(current_dir)[0])[0] # go up twice to go up to thesis_sourcecode
    
    return abs_path
    
@make_output_path_obj
def get_abs_raw_data_folder_path():
    return os.path.join(get_abs_path_of_package_root(),"data/csv_files/raw")

@make_output_path_obj
def get_abs_saved_models_folder_path():
    return os.path.join(get_abs_path_of_package_root(),"src/model_training/saved_models")


@make_output_path_obj
def get_abs_saved_patch_models_folder_path_with_model_full_name(name):
    return os.path.join(get_abs_saved_models_folder_path(),name + ".keras")

@make_output_path_obj
def get_abs_saved_patch_models_folder_path_with_model_name(name):
    model_names = os.listdir(get_abs_saved_models_folder_path())
    
    for i in model_names:
        if name in i:
            return os.path.join(get_abs_saved_models_folder_path(),i)
        
@make_output_path_obj
def get_abs_saved_surface_points_models_folder_path_with_model_name(name):
    model_names = os.listdir(get_abs_saved_models_folder_path())
    for i in model_names:
        # print(i)
        if name in i:
            return os.path.join(get_abs_saved_models_folder_path(),i)
        
@make_output_path_obj
def get_abs_saved_signed_distance_models_folder_path_with_model_name(name):
    model_names = os.listdir(get_abs_saved_models_folder_path())
    for i in model_names:
        # print(i)
        if name in i:
            return os.path.join(get_abs_saved_models_folder_path(),i)
        
@make_output_path_obj
def get_abs_saved_signed_distance_models_folder_path_with_model_name(name):
    model_names = os.listdir(get_abs_saved_models_folder_path())
    for i in model_names:
        # print(i)
        if name in i:
            return os.path.join(get_abs_saved_models_folder_path(),i)
        

def get_list_of_elements_in_dir(dir_abs_path):
    return [f for f in Path(dir_abs_path).iterdir() if f.is_file()]

@make_output_path_obj
def get_abs_path(file_name): 
    return os.path.abspath(file_name)

@make_output_path_obj
def get_abs_saved_plots_folder_path():
    return os.path.join(get_abs_path_of_package_root(),'data/plots')

@make_output_path_obj
def get_abs_patch_model_plots_folder_path():
    return os.path.join(get_abs_path_of_package_root(),"data/plots/patch_model")

@make_output_path_obj
def get_abs_model_training_data_folder_path():
    return os.path.join(get_abs_path_of_package_root(),'data/csv_files/model_training_history')

@make_output_path_obj
def get_patch_model_training_data_folder_path():
    return os.path.join(get_abs_model_training_data_folder_path(), "patch_model")

@make_output_path_obj
def get_surface_points_model_training_data_folder_path():
    return os.path.join(get_abs_model_training_data_folder_path(), "surface_points_model")
@make_output_path_obj
def get_signed_distance_model_training_data_folder_path():
    return os.path.join(get_abs_model_training_data_folder_path(), "signed_distance_model")

@make_output_path_obj
def get_patch_model_training_data_file_abs_path_by_model_name(name):
    model_names_traning_history = os.listdir(get_patch_model_training_data_folder_path())
    # print("model history trainings: ", model_names_traning_history)
    for i in model_names_traning_history:
        if name in i:
            return os.path.join(get_patch_model_training_data_folder_path(),i)

@make_output_path_obj
def get_surface_points_model_training_data_file_abs_path_by_model_name(name):
    model_names_traning_history = os.listdir(get_surface_points_model_training_data_folder_path())
    # print("model history trainings: ", model_names_traning_history)
    for i in model_names_traning_history:
        if name in i:
            return os.path.join(get_surface_points_model_training_data_folder_path(),i)
@make_output_path_obj
def get_signed_distance_model_training_data_file_abs_path_by_model_name(name):
    model_names_traning_history = os.listdir(get_signed_distance_model_training_data_folder_path())
    # print("model history trainings: ", model_names_traning_history)
    for i in model_names_traning_history:
        if name in i:
            return os.path.join(get_signed_distance_model_training_data_folder_path(),i)

@make_output_path_obj
def get_plot_name_and_abs_path_from_model_training_data_file_name(name, ext="jpg"):
    model_names_traning_history = os.listdir(get_patch_model_training_data_folder_path())
    # print("model history trainings: ", model_names_traning_history)
    plot_name = ""
    plot_path = get_abs_patch_model_plots_folder_path()
    try:
        for i in model_names_traning_history:
            if name in i:
                plot_name = i.replace("csv", ext)


                raise StopIteration
    except StopIteration:
        
        plot_abs_path_and_name = os.path.join(plot_path, plot_name)

        return plot_abs_path_and_name