
def add_L2_regularization_to_output_layer_for_loaded_patch_model(model):
    loaded_model = model
    loaded_model.add_L2_regularizer_to_output_layer()


def add_L2_regularization_to_output_layer_by_creating_a_new_model(patch_model, PatchModelUtils):
    new_patch_model_with_regularizer = PatchModelUtils.add_L2_regularizer_to_output_layer_by_creating_a_new_model(patch_model)


