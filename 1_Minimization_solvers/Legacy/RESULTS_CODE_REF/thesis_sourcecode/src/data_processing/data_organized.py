import os
import pandas as pd
import tensorflow as tf
from ..utils.path_funcs import *
from matplotlib import pyplot as plt
# from importlib_resources import contents

def collect_csv_files_into_one_df(edges_only=False):
    # data_csv_files_path = os.path.join(get_abs_path(get_relative_raw_data_folder_path()),"raw/")
    data_csv_files_path = get_abs_raw_data_folder_path()
    all_csv_files = get_list_of_elements_in_dir(data_csv_files_path)
    if edges_only:
        all_csv_files = [i for i in get_list_of_elements_in_dir(data_csv_files_path) if "Edges" in str(i)]
    print("number of files: ", len(all_csv_files))
    dfs = []
    col_names = ["x", "y", "z", "patch", "sp1", "sp2", "sd"]
    # READY_CSV = pd.read_csv(os.path.join(data_csv_files_path, all_csv_files[0]),  names=col_names)
    # print(READY_CSV['patch'].unique())
    uniques = []
    total_size = 0
    for i in all_csv_files:
        # path to file Points_0_0_0.csv
        path = os.path.join(data_csv_files_path, i)#all_csv_files[-1])
        READY_CSV = pd.read_csv(path, names=col_names, engine="pyarrow")
        total_size+= len(READY_CSV.index)
        # print("number of rows: ", len(READY_CSV.index), "total: ", total_size)
        dfs.append(READY_CSV)
        
        # for i in READY_CSV['patch'].unique():
        #     uniques.append(i)
        # print(READY_CSV['patch'].unique())

    # print(min(uniques), max(uniques))
    # uniques = list(set(uniques))
    # uniques.sort()
    # print(uniques)
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.sort_values(by="patch")
    # correction of some data points
    # print("final_df.shape before filter: ", final_df.shape)
    # final_df = final_df[(final_df['sp1']>=0)&(final_df['sp1']<=1)&(final_df['sp2']>=0)&(final_df['sp2']<=1)]
    # print("final_df.shape AFTER filter: ", final_df.shape)
    # print("final_df.shape", final_df.shape)
    print(final_df.head())
    # print(final_df.tail())
    return final_df

def split_df_based_on_patch(df):
    dfs = []
    for _, d in df.groupby('patch'):
        dfs.append(d)
    
    return dfs

def split_dfs_for_training_testing_and_recombine(dfs, amount=960000, split=0.2, random_state=1):
    train = []
    test = []
    
    for df in dfs:

        length = df.shape[0]
       
        if amount > 0:
            num = amount//len(dfs)
            if type(amount) == int:
                df = df.sample(n=num,random_state=random_state)
            elif type(amount) == float:
                df = df.sample(frac=amount,random_state=random_state)

            # limit_amount = int(length*amount)
            # df = df.iloc[:limit_amount]
            length = df.shape[0]
        limit_split = int(length*split)

        training_df = df.iloc[limit_split:,:]
        train.append(training_df)

        testing_df = df.iloc[:limit_split,:]
        test.append(testing_df)

    train_final = pd.concat(train, ignore_index=True)
    
    print(f"training samples: {train_final.shape}")
    test_final = pd.concat(test, ignore_index=True)
    print(f"testing samples: {test_final.shape}")
    return train_final, test_final

def get_xys_sp1_sp2_sd_from_df(df):
    # print("df = ", df)
    x = df["x"].tolist() 
    y = df["y"].tolist() 
    z = df["z"].tolist()
    patch = df["patch"].tolist()
    sp1 = df["sp1"].tolist() 
    sp2 = df["sp2"].tolist() 
    sd = df["sd"].tolist()
    return x, y, z, patch, sp1, sp2, sd

# print(get_xys_sp1_sp2_sd_from_df(READY_CSV))

def create_patch_model_training_data(df, amount=960000,split=0.2,random_state=1):
    
    dfs = split_df_based_on_patch(df)
    training_df, testing_df = split_dfs_for_training_testing_and_recombine(dfs,amount=amount, split=split, random_state=random_state)


    x_trn, y_trn, z_trn, patch_trn, sp1_trn, sp2_trn, sd_trn = get_xys_sp1_sp2_sd_from_df(training_df)
    X_train = [(x_trn[i], y_trn[i], z_trn[i]) for i in range(training_df.shape[0])]
    Y_train = [patch_trn[i] for i in range(training_df.shape[0])]
    x_tst, y_tst, z_tst, patch_tst, sp1_tst, sp2_tst, sd_tst = get_xys_sp1_sp2_sd_from_df(testing_df)
    X_test = [(x_tst[i], y_tst[i], z_tst[i]) for i in range(testing_df.shape[0])]
    Y_test = [patch_tst[i] for i in range(testing_df.shape[0])]
    
    
    return X_train, X_test, Y_train, Y_test


def create_signed_distance_model_training_data(df, amount=960000,split=0.2,random_state=1):
    
    dfs = split_df_based_on_patch(df)
    training_df, testing_df = split_dfs_for_training_testing_and_recombine(dfs,amount=amount, split=split, random_state=random_state)


    x_trn, y_trn, z_trn, patch_trn, sp1_trn, sp2_trn, sd_trn = get_xys_sp1_sp2_sd_from_df(training_df)
    X_train = [(x_trn[i], y_trn[i], z_trn[i]) for i in range(training_df.shape[0])]
    Y_train = [sd_trn[i] for i in range(training_df.shape[0])]
    x_tst, y_tst, z_tst, patch_tst, sp1_tst, sp2_tst, sd_tst = get_xys_sp1_sp2_sd_from_df(testing_df)
    X_test = [(x_tst[i], y_tst[i], z_tst[i]) for i in range(testing_df.shape[0])]
    Y_test = [sd_tst[i] for i in range(testing_df.shape[0])]
    
    
    return X_train, X_test, Y_train, Y_test

def create_surface_points_model_training_data(df, amount=960000,split=0.2):
    
    dfs = split_df_based_on_patch(df)
    training_df, testing_df = split_dfs_for_training_testing_and_recombine(dfs,amount=amount, split=split)


    # training_df = training_df.sample(frac=1)
    x_trn, y_trn, z_trn, patch_trn, sp1_trn, sp2_trn, sd_trn = get_xys_sp1_sp2_sd_from_df(training_df)
    X_train_points = [(x_trn[i], y_trn[i], z_trn[i]) for i in range(training_df.shape[0])]
    X_train_patches = [patch_trn[i] for i in range(training_df.shape[0])]
    Y_train_p1 = [sp1_trn[i] for i in range(training_df.shape[0])]
    Y_train_p2 = [sp2_trn[i] for i in range(training_df.shape[0])]
    x_tst, y_tst, z_tst, patch_tst, sp1_tst, sp2_tst, sd_tst = get_xys_sp1_sp2_sd_from_df(testing_df)
    X_test_points = [(x_tst[i], y_tst[i], z_tst[i]) for i in range(testing_df.shape[0])]
    X_test_patches = [patch_tst[i] for i in range(testing_df.shape[0])]
    Y_test_p1 = [sp1_tst[i] for i in range(testing_df.shape[0])]
    Y_test_p2 = [sp2_tst[i] for i in range(testing_df.shape[0])]
    
    
    return X_train_points,X_train_patches, X_test_points,X_test_patches, Y_train_p1, Y_train_p2, Y_test_p1,Y_test_p2

def create_surface_points_model_training_data_for_one_patch(patch, df, amount=960000,split=0.2, random_state=1):
    dfs = split_df_based_on_patch(df)
    primary_df = pd.DataFrame({})
    for i in dfs:
        try:
            print(f"now trying to find df with patch {patch}. current df's patch: {i['patch'].iloc[0]}")
            
            if i["patch"].iloc[0] == patch:
                primary_df = i
                raise StopIteration
        except StopIteration:
            print(f"now processing df with patch {primary_df['patch'].iloc[0]}")
            break
    
    training_df, testing_df = split_dfs_for_training_testing_and_recombine([primary_df],amount=amount, split=split, random_state=random_state)

    scaling = 0
    # training_df = training_df.sample(frac=1)
    x_trn, y_trn, z_trn, patch_trn, sp1_trn, sp2_trn, sd_trn = get_xys_sp1_sp2_sd_from_df(training_df)
    X_train_points = [(x_trn[i], y_trn[i], z_trn[i]) for i in range(training_df.shape[0])]
    # X_train_patches = [patch_trn[i] for i in range(training_df.shape[0])]
    # Y_train_p1 = [sp1_trn[i] for i in range(training_df.shape[0])]
    # Y_train_p2 = [sp2_trn[i] for i in range(training_df.shape[0])]
    Y_train_sp = [(sp1_trn[i]*(10**scaling),sp2_trn[i]*(10**scaling)) for i in range(training_df.shape[0])]
    x_tst, y_tst, z_tst, patch_tst, sp1_tst, sp2_tst, sd_tst = get_xys_sp1_sp2_sd_from_df(testing_df)
    X_test_points = [(x_tst[i], y_tst[i], z_tst[i]) for i in range(testing_df.shape[0])]
    # X_test_patches = [patch_tst[i] for i in range(testing_df.shape[0])]
    # Y_test_p1 = [sp1_tst[i] for i in range(testing_df.shape[0])]
    # Y_test_p2 = [sp2_tst[i] for i in range(testing_df.shape[0])]
    Y_test_sp = [(sp1_tst[i]*(10**scaling),sp2_tst[i]*(10**scaling)) for i in range(testing_df.shape[0])]

    return X_train_points, Y_train_sp, X_test_points, Y_test_sp 

def get_training_and_testing_data_for_patch_model(amount=960000, split=0.2,random_state=1, edges_only=False):
    df = collect_csv_files_into_one_df(edges_only=edges_only)
    X_train, X_test, Y_train, Y_test = create_patch_model_training_data(df, amount=amount, split=split,random_state=random_state)
    X_train, X_test, Y_train, Y_test = tf.convert_to_tensor(X_train),tf.convert_to_tensor(X_test),tf.convert_to_tensor(Y_train),tf.convert_to_tensor(Y_test)
    return X_train, X_test, Y_train, Y_test

def get_training_and_testing_data_and_sample_weights_for_patch_model(amount=960000, split=0.2,random_state=1):
    df = collect_csv_files_into_one_df()
    total_number_of_points = len(df.index)
    dfs = split_df_based_on_patch(df)
    sample_weights_per_patch = {}
    # weight for a patch: total_number_of_points/number_of_points_in_a_patch
    # sum of all weights, divide every weight by the sum
    print("total number of points: ", total_number_of_points)
    for i in dfs:
        patch = i["patch"].iloc[0]
        sample_weights_per_patch[patch] = total_number_of_points/len(i.index)
    sum_of_weights = sum(sample_weights_per_patch.values())
    for key, value in sample_weights_per_patch.items():
        sample_weights_per_patch[key] = value/sum_of_weights
        print(f"for patch: {key}, number of samples: {len(dfs[key].index)}| training weight: {sample_weights_per_patch[key]}")

    print("sum of probabilities of training weights: ", sum(sample_weights_per_patch.values()))
    X_train, X_test, Y_train, Y_test = create_patch_model_training_data(df, amount=amount, split=split,random_state=random_state)
    
    sample_weights_for_training_data = [sample_weights_per_patch[i] for i in Y_train]
    sample_weights_for_training_data =  pd.Series(sample_weights_for_training_data).to_frame()
    print("first 3 patches: ", Y_train[:3])
    print("first 3 patches weights: ", sample_weights_for_training_data[:3])
    return X_train, X_test, Y_train, Y_test, sample_weights_for_training_data
    

def get_training_and_testing_data_for_surface_point_model_for_one_patch(patch, amount=960000,split=0.2, random_state=1):
    df = collect_csv_files_into_one_df()
    X_train_points, Y_train_sp, X_test_points, Y_test_sp  = create_surface_points_model_training_data_for_one_patch(patch=patch, df=df, amount=amount,split=split ,random_state=random_state)
    X_train_points, Y_train_sp, X_test_points, Y_test_sp  = tf.convert_to_tensor(X_train_points),tf.convert_to_tensor(Y_train_sp),tf.convert_to_tensor(X_test_points),tf.convert_to_tensor(Y_test_sp)

    return X_train_points, Y_train_sp, X_test_points, Y_test_sp  


def get_training_and_testing_data_for_sp_model(amount=960000, split=0.2):
        df = collect_csv_files_into_one_df()

        X_train_points, X_train_patches,
        X_test_points,X_test_patches,
        Y_train_p1, Y_train_p2,
        Y_test_p1,Y_test_p2 = create_surface_points_model_training_data(df, amount=amount, split=split)

        X_train_points = tf.convert_to_tensor(X_train_points)
        X_train_patches = tf.convert_to_tensor(X_train_patches)
        X_test_points = tf.convert_to_tensor(X_test_points)
        X_test_patches = tf.convert_to_tensor(X_test_patches)
        Y_train_p1 = tf.convert_to_tensor(Y_train_p1)
        Y_train_p2 = tf.convert_to_tensor(Y_train_p2)
        Y_test_p1 = tf.convert_to_tensor(Y_test_p1)
        Y_test_p2 = tf.convert_to_tensor(Y_test_p2)
       
        return X_train_points,X_train_patches, X_test_points,X_test_patches, Y_train_p1, Y_train_p2, Y_test_p1,Y_test_p2


def get_training_and_testing_data_for_signed_distance_model(amount=960000, split=0.2,random_state=1):
    df = collect_csv_files_into_one_df()
    X_train, X_test, Y_train, Y_test = create_signed_distance_model_training_data(df, amount=amount, split=split,random_state=random_state)
    X_train, X_test, Y_train, Y_test = tf.convert_to_tensor(X_train),tf.convert_to_tensor(X_test),tf.convert_to_tensor(Y_train),tf.convert_to_tensor(Y_test)
    return X_train, X_test, Y_train, Y_test


def plot_data(training_data, testing_data):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    
    x, y, z = zip(*training_data)
    xt, yt, zt = zip(*testing_data)

    ax.scatter(x, y, z, c='tab:blue', label="Training Points")
    ax.scatter(xt, yt, zt, c='tab:orange', label="Testing Points")
    ax.legend()
    plt.show()
# def sample_random_points_from_data_set_per_patch():
#     df = collect_csv_files_into_one_df()
#     dfs = split_df_based_on_patch(df)
#     training_df, testing_df = split_dfs_for_training_testing_and_recombine(dfs,amount=amount, split=split)

def plot_bar_points_per_patch(data):
    patches = [i for i in range(96)]
    counts = [len(data[data["patch"]==i]) for i in patches]
    # for i in patches:
    #     print(f"patch: {i}, count: {counts[i]}")
    fig, ax = plt.subplots()
    ax.set_ylabel('Number of Points')
    ax.set_xlabel('Patch')
    ax.set_xticks([i for i in range(0, 100, 5)])
    ax.set_xlim(left=- 1.0, right=96.0)
    ax.set_title('Number of Points per patch')
    ax.bar(patches, counts)
    plt.show()



def get_N_random_points_per_patch_for_patch_model_training(N, random_state=1, split=0.2):
    df = collect_csv_files_into_one_df()
    dfs = split_df_based_on_patch(df)
    sampled_dfs = []
    for each_df in dfs:
        each_df = each_df.sample(n = N, random_state=random_state)
        sampled_dfs.append(each_df)

    training_df, testing_df = split_dfs_for_training_testing_and_recombine(sampled_dfs,amount=960000, split=split)
    plot_bar_points_per_patch(training_df)
    plot_bar_points_per_patch(testing_df)
    x_trn, y_trn, z_trn, patch_trn, sp1_trn, sp2_trn, sd_trn = get_xys_sp1_sp2_sd_from_df(training_df)
    X_train = [(x_trn[i], y_trn[i], z_trn[i]) for i in range(training_df.shape[0])]
    Y_train = [patch_trn[i] for i in range(training_df.shape[0])]
    x_tst, y_tst, z_tst, patch_tst, sp1_tst, sp2_tst, sd_tst = get_xys_sp1_sp2_sd_from_df(testing_df)
    X_test = [(x_tst[i], y_tst[i], z_tst[i]) for i in range(testing_df.shape[0])]
    Y_test = [patch_tst[i] for i in range(testing_df.shape[0])]
    
    X_train, X_test, Y_train, Y_test = tf.convert_to_tensor(X_train),tf.convert_to_tensor(X_test),tf.convert_to_tensor(Y_train),tf.convert_to_tensor(Y_test)

    return X_train, X_test, Y_train, Y_test


