import numpy as np
import pickle
from constants import config, sar_fill_value, smap_fill_value, smap_fill_value_scaled, smap_valid_max
# TODO: Consider moving these operations to package_data.py


#
# Open datasets.
#
def open_datasets():
    with open(config.path_root + 'sar_train.pickle', 'rb') as f:
        sar_train = pickle.load(f)
    with open(config.path_root + 'smap_train.pickle', 'rb') as f:
        smap_train = pickle.load(f)
    with open(config.path_root + 'sar_test.pickle', 'rb') as f:
        sar_test = pickle.load(f)
    with open(config.path_root + 'smap_test.pickle', 'rb') as f:
        smap_test = pickle.load(f)
    with open(config.path_root + 'sar_train_titles.pickle', 'rb') as f:
        sar_train_titles = pickle.load(f)
    with open(config.path_root + 'smap_train_titles.pickle', 'rb') as f:
        smap_train_titles = pickle.load(f)
    with open(config.path_root + 'sar_test_titles.pickle', 'rb') as f:
        sar_test_titles = pickle.load(f)
    with open(config.path_root + 'smap_test_titles.pickle', 'rb') as f:
        smap_test_titles = pickle.load(f)
    return sar_train, smap_train, sar_test, smap_test, \
           sar_train_titles, smap_train_titles, sar_test_titles, smap_test_titles


#
# Pre-processing
#
sar_train, smap_train, sar_test, smap_test, \
    sar_train_titles, smap_train_titles, sar_test_titles, smap_test_titles = open_datasets()

if 'Fill_Value_Samples_Excluded' not in config.path_root:
    # Mask and normalize data containing fill values
    sar_train_mask = sar_train == sar_fill_value
    smap_train_mask = np.logical_or(smap_train == smap_fill_value, smap_train > smap_valid_max)
    sar_train_masked = sar_train
    sar_train_masked[smap_train_mask] = sar_fill_value
    sar_train_masked = np.ma.masked_where(sar_train_masked == sar_fill_value, sar_train_masked)

    masked_mean = np.ma.mean(sar_train_masked, axis=0)
    masked_std = np.ma.std(sar_train_masked, axis=0)

    # Training Sets
    input_shape = sar_train_masked.shape[1:]
    train_samples, height, width, channels = sar_train_masked.shape
    sar_train_masked_normed = (sar_train_masked - masked_mean) / masked_std
    sar_train_masked_normed = sar_train_masked_normed.data
    smap_train[smap_train_mask] = smap_fill_value_scaled

    # Testing Sets
    sar_test_mask = sar_test == sar_fill_value
    smap_test_mask = np.logical_or(smap_test == smap_fill_value, smap_test > smap_valid_max)
    sar_test_masked = sar_test
    sar_test_masked[smap_test_mask] = sar_fill_value
    sar_test_masked = np.ma.masked_where(sar_test_masked == sar_fill_value, sar_test_masked)
    input_shape = sar_test_masked.shape[1:]
    test_samples, height, width, channels = sar_test_masked.shape
    sar_test_masked_normed = (sar_test_masked - masked_mean) / masked_std
    sar_test_masked_normed = sar_test_masked_normed.data
    smap_test[smap_test_mask] = smap_fill_value_scaled

    var_map = {
        'sar_train': sar_train,
        'sar_train_masked_normed': sar_train_masked_normed,
        'sar_test': sar_test,
        'sar_test_masked_normed': sar_test_masked_normed,
        'smap_train': smap_train,
        'smap_train_mask': smap_train_mask,
        'smap_test': smap_test,
        'smap_test_mask': smap_test_mask,
        'input_shape': input_shape,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'width': width,
        'height': height,
        'channels': channels
    }
else:
    # Normalize Data that does not contain fill values
    mean = np.mean(sar_train, axis=0)
    std = np.std(sar_train, axis=0)

    # Training Sets
    input_shape = sar_train.shape[1:]
    train_samples, height, width, channels = sar_train.shape
    sar_train_flat = sar_train.reshape((train_samples, width*height))
    sar_train_normed = (sar_train - mean) / std
    sar_train_normed_flat = sar_train_normed.reshape((train_samples, width*height))
    smap_train_flat = smap_train.reshape((train_samples, width*height))

    # Testing Sets
    input_shape_flat = sar_train_flat.shape[1:]
    test_samples, height, width, channels = sar_test.shape
    sar_test_flat = sar_test.reshape((test_samples, width*height))
    sar_test_normed = (sar_test - mean) / std
    sar_test_normed_flat = sar_test_normed.reshape((test_samples, width*height))
    smap_test_flat = smap_test.reshape((test_samples, width*height))

    var_map = {
        'sar_train': sar_train,
        'sar_train_flat': sar_train_flat,
        'sar_train_normed': sar_train_normed,
        'sar_train_normed_flat': sar_train_normed_flat,
        'sar_test': sar_test,
        'sar_test_flat': sar_test_flat,
        'sar_test_normed': sar_test_normed,
        'sar_test_normed_flat': sar_test_normed_flat,
        'smap_train': smap_train,
        'smap_train_flat': smap_train_flat,
        'smap_test': smap_test,
        'smap_test_flat': smap_test_flat,
        'input_shape': input_shape,
        'input_shape_flat': input_shape_flat,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'width': width,
        'height': height,
        'channels': channels
    }
