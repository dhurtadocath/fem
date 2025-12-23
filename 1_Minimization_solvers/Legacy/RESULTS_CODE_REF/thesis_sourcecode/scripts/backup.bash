cp ../data/csv_files/model_training_history/patch_model/*.csv ../../backup/training_history/patch_model/
cp ../src/model_training/saved_models/*.keras ../../backup/models/patch_classification_models
#backup surface points regression models data
cp ../data/csv_files/model_training_history/surface_points_model*.csv ../../backup/training_history/surface_points_models/
cp ../src/model_training/saved_models/*.keras ../../backup/models