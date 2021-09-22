import os
import sys
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from textwrap import wrap
from constants import config, config_file_name, smap_fill_value_scaled, sar_fill_value
from pre_processing import var_map, sar_train_titles, smap_train_titles, smap_test_titles, sar_test_titles
from metric_and_model_definitions import func_map

# Uncomment the lines below and run keras_model.py in the terminal to enter the CLI debug wrapper session.
# import tensorflow as tf
# from tensorflow.python import debug as tf_debug
# tf.keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
# epochs = 1

# Disable the warning: "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Limit TensorFlow's automatic reservation of the entire GPU memory.
cp = tf.ConfigProto()
cp.gpu_options.allow_growth = True
K.set_session(tf.Session(config=cp))


#
# Printing, Plotting, & Image Visualization Definitions.
#
def print_results(title, results, config):
    print(title)
    print(config.loss_name, results[0])
    for metric_name, metric_result in zip(config.metric_names, results[1:]):
        print(metric_name, metric_result)
    print()


def print_to_file():
    original_stdout = sys.stdout
    with open(path_results + 'console_output.txt', 'wt') as output_log:
        sys.stdout = output_log
        print('Model Summary:')
        model.summary()
        print()
        print_results('Training Results:', results_train, config)
        print_results('Testing Results:', results_test, config)
    sys.stdout = original_stdout


def plot_history(dict):
    fig = plt.figure()
    plt.plot(dict['loss'], color='b')
    plt.plot(dict['val_loss'], color='r')
    plt.title('Loss vs. epochs')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(['Training Loss', 'Testing Loss'])
    fig.savefig(path_results + 'loss_history.png')
    plt.pause(0.5)
    plt.close(fig)


def vis_pred(str, sar_set, smap_set, pred_set, sar_set_titles, smap_set_titles):
    def colorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

    for i in range(10):
        sar_sample = sar_set[i].reshape((var_map['height'], var_map['width']))
        smap_sample = smap_set[i].reshape((var_map['height'], var_map['width']))
        pred_sample = pred_set[i].reshape((var_map['height'], var_map['width']))
        if 'Fill_Value_Samples_Excluded' not in config.path_root:
            smap_mask = smap_sample == smap_fill_value_scaled
            pred_mask = pred_sample < 0
            if smap_sample[~smap_mask].size == 0 and pred_sample[~pred_mask].size == 0:
                # This occurs when the entire SMAP image is a fill value but I still want to visualize the prediction.
                vmin = np.min(pred_sample)
                vmax = np.max(pred_sample)
            else:
                vmin = np.min([np.min(smap_sample[~smap_mask]), np.min(pred_sample[~pred_mask])])
                vmax = np.max([np.max(smap_sample[~smap_mask]), np.max(pred_sample[~pred_mask])])
        else:
            vmin = np.min([np.min(smap_sample), np.min(pred_sample)])
            vmax = np.max([np.max(smap_sample), np.max(pred_sample)])

        fig = plt.figure()
        plt.subplot(221)
        if 'Fill_Value_Samples_Excluded' not in config.path_root:
            masked_sar = np.ma.masked_where(sar_sample == sar_fill_value, sar_sample)
            cmap = mpl.cm.viridis
            cmap.set_bad(color='black')
            sar_img = plt.imshow(masked_sar, cmap=cmap)
        else:
            sar_img = plt.imshow(sar_sample)
        if 'Max_Crop_Size' in config.path_root:
            sar_title = '\n'.join(wrap('_'.join(sar_set_titles[i].split('-')[:6]), 30))
        else:
            sar_title = '\n'.join(wrap('_'.join(sar_set_titles[i].split('-')[:6]) + '_' + sar_set_titles[i][-7:-4], 30))
        plt.title(f'SAR {str} {i}:\n' + sar_title, fontsize=8)
        colorbar(sar_img)

        plt.subplot(222)
        if 'Fill_Value_Samples_Excluded' not in config.path_root:
            masked_smap = np.ma.masked_where(smap_sample == smap_fill_value_scaled, smap_sample)
            cmap = mpl.cm.viridis
            cmap.set_bad(color='black')
            smap_img = plt.imshow(masked_smap, cmap=cmap)
        else:
            smap_img = plt.imshow(smap_sample)
        if 'Max_Crop_Size' in config.path_root:
            smap_title = '\n'.join(wrap(smap_set_titles[i][:52], 37))
        else:
            smap_title = '\n'.join(wrap(smap_set_titles[i][:52] + smap_set_titles[i][-8:-4], 37))
        plt.title(f'SMAP {str} {i}:\n' + smap_title, fontsize=8)
        colorbar(smap_img)
        plt.clim(vmin=vmin, vmax=vmax)

        plt.subplot(224)
        if 'Fill_Value_Samples_Excluded' not in config.path_root:
            pred_mask = pred_sample < 0
            if pred_sample[~pred_mask].size == 0:
                # This occurs when the entire SMAP image is a fill value but I still want to visualize the prediction.
                pred_img = plt.imshow(pred_sample)
            elif 'masked' in config.loss:
                masked_pred = np.ma.masked_where(smap_sample == smap_fill_value_scaled, pred_sample)
                cmap = mpl.cm.viridis
                cmap.set_bad(color='black')
                pred_img = plt.imshow(masked_pred, cmap=cmap)
            else:
                masked_pred = np.ma.masked_where(pred_sample < 0, pred_sample)
                cmap = mpl.cm.viridis
                cmap.set_bad(color='black')
                pred_img = plt.imshow(masked_pred, cmap=cmap)
        else:
            pred_img = plt.imshow(pred_sample)
        if 'Max_Crop_Size' in config.path_root:
            pred_title = '\n'.join(wrap(smap_set_titles[i][:52], 37))
        else:
            pred_title = '\n'.join(wrap(smap_set_titles[i][:52] + smap_set_titles[i][-8:-4], 37))
        plt.title(f'Prediction {str} {i}:\n' + pred_title, fontsize=8)
        colorbar(pred_img)
        plt.clim(vmin=vmin, vmax=vmax)

        plt.tight_layout()
        if 'Max_Crop_Size' in config.path_root:
            fig_file_name = 'PRED_' + str.upper() + '_' + smap_set_titles[i][:52]
        else:
            fig_file_name = 'PRED_' + str.upper() + '_' + smap_set_titles[i][:52] + smap_set_titles[i][-8:-4]

        fig.savefig(path_results + fig_file_name + '.png')
        plt.pause(0.01)
        plt.close(fig)


#
# Adding configuration capabilities to the file.
#
path_results = 'results/results_' + config_file_name.split('_')[-1] + '/'
try:
    os.makedirs(path_results)
except FileExistsError:
    print('Directory:', path_results, 'already exists.')
else:
    print('Directory:', path_results, 'created.')


#
# Run the model.
#
model = func_map[config.model]()
model.summary()
# TODO Look into how to feed TF dataset from package_data.py into Keras model.
model.compile(loss=func_map[config.loss], optimizer=optimizers.Adam(),
              metrics=[func_map[metric] for metric in config.metrics])
callbacks = [
    EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=config.reduce_lr_patience, min_lr=0.00001, verbose=1),
    ModelCheckpoint(filepath=path_results + 'model_best.h5', monitor='val_loss', verbose=1, save_best_only=True)
]
history = model.fit(
    var_map[config.x_train],
    var_map[config.y_train],
    batch_size=config.batch_size,
    epochs=config.epochs,
    verbose=1,
    validation_data=(var_map[config.x_test], var_map[config.y_test]),
    callbacks=callbacks
)
hist_dict = history.history
hist_dict['lr'] = [np.float64(x) for x in hist_dict['lr']]


#
# Save the model.
#
model.save(path_results + 'model.h5')  # Saves the architecture, the trained weights, and what's passed to compile.
model_json = model.to_json()  # Returns a JSON formatted string of the architecture only.
with open(path_results + 'model.json', 'wt') as f:
    json.dump(json.loads(model_json), f, indent=4)

weights = model.get_weights()
model.save_weights(path_results + 'model_weights.h5')  # Saves the weights only.

model_config = model.get_config()  # Returns a dictionary of the architecture only.
with open(path_results + 'model_config.json', 'wt') as f:
    json.dump(model_config, f, indent=4)

with open(path_results + 'history.json', 'wt') as f:
    json.dump(hist_dict, f, indent=4)


#
# Results
#
plot_history(hist_dict)
results_train = model.evaluate(var_map[config.x_train], var_map[config.y_train], batch_size=config.batch_size)
results_test = model.evaluate(var_map[config.x_test], var_map[config.y_test], batch_size=config.batch_size)
print_results('Training Results:', results_train, config)
print_results('Testing Results:', results_test, config)


#
# Visualize predictions.
#
pred_train = model.predict(var_map[config.x_train])
pred_test = model.predict(var_map[config.x_test])
vis_pred('train', var_map['sar_train'], var_map['smap_train'], pred_train, sar_train_titles, smap_train_titles)
vis_pred('test', var_map['sar_test'], var_map['smap_test'], pred_test, sar_test_titles, smap_test_titles)

print_to_file()
print('Program Terminated')
