import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import models
import matplotlib.pyplot as plt
from constants import config, config_file_name, smap_fill_value_scaled
from pre_processing import var_map
from metric_and_model_definitions import func_map


# Disable the warning: "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Limit TensorFlow's automatic reservation of the entire GPU memory.
cp = tf.ConfigProto()
cp.gpu_options.allow_growth = True
K.set_session(tf.Session(config=cp))


def r_squared_numpy(y_true, y_pred):
    n = var_map['test_samples'] * var_map['width'] * var_map['height']
    a1 = (n*np.sum(y_true*y_pred) - np.sum(y_true)*np.sum(y_pred)) / \
         (n*np.sum(np.square(y_true)) - np.square(np.sum(y_true)))
    a0 = np.mean(y_pred) - a1*np.mean(y_true)
    y_line = a0 + a1*y_true
    y_mean = np.mean(y_pred)
    ss_tot = np.sum(np.square(y_pred - y_mean))
    ss_res = np.sum(np.square(y_pred - y_line))
    return 1 - ss_res / ss_tot


def confusion(true, pred):
    matrix = np.zeros((2, 2), dtype=int)
    tp_true = true.flatten()[np.logical_and(pred.flatten() > 0, true.flatten() > 0)]
    tp_pred = pred.flatten()[np.logical_and(pred.flatten() > 0, true.flatten() > 0)]
    matrix[0, 0] = len(tp_true)
    fp_true = true.flatten()[np.logical_and(pred.flatten() > 0, true.flatten() < 0)]
    fp_pred = pred.flatten()[np.logical_and(pred.flatten() > 0, true.flatten() < 0)]
    matrix[0, 1] = len(fp_true)
    fn_true = true.flatten()[np.logical_and(pred.flatten() < 0, true.flatten() > 0)]
    fn_pred = pred.flatten()[np.logical_and(pred.flatten() < 0, true.flatten() > 0)]
    matrix[1, 0] = len(fn_true)
    tn_true = true.flatten()[np.logical_and(pred.flatten() < 0, true.flatten() < 0)]
    tn_pred = pred.flatten()[np.logical_and(pred.flatten() < 0, true.flatten() < 0)]
    matrix[1, 1] = len(tn_true)
    return matrix, tp_true, tn_true, fn_true, fp_true, tp_pred, tn_pred, fn_pred, fp_pred


def confusion_stats(matrix):
    tp, fp, fn, tn = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    TPR = tp / (tp + fn)  # True Positive Rate
    TNR = tn / (tn + fp)  # True Negative Rate
    FPR = fp / (tn + fp)  # False Positive Rate
    FNR = fn / (tp + fn)  # False Negative Rate
    PPV = tp / (tp + fp)  # Positive Predictive Value
    NPV = tn / (tn + fn)  # Negative Predictive Value
    FDR = fp / (tp + fp)  # False Discovery Rate
    FOR = fn / (tn + fn)  # False Omission Rate
    ACC = (tp + tn) / (tp + tn + fp + fn)
    return [[TPR, FPR], [FNR, TNR]], [[PPV, FDR], [FOR, NPV]], ACC


def calculate_correlations():
    # np.corrcoef returns a matrix: [[corr(x, x), corr(x, y)], [corr(y, x), corr(y, y)]]
    if 'Fill_Value_Samples_Excluded' not in config.path_root:
        if 'masked' in config.loss:
            mask_train = var_map['smap_train'] == smap_fill_value_scaled
            mask_test = var_map['smap_test'] == smap_fill_value_scaled
            corr_train = np.corrcoef(var_map['smap_train'][~mask_train].flatten(),
                                     pred_train[~mask_train].flatten())[0, 1]
            corr_test = np.corrcoef(var_map['smap_test'][~mask_test].flatten(),
                                    pred_test[~mask_test].flatten())[0, 1]
        else:
            corr_train = np.corrcoef(tp_smap_train, tp_pred_train)[0, 1]
            corr_test = np.corrcoef(tp_smap_test, tp_pred_test)[0, 1]
    else:
        corr_train = np.corrcoef(var_map['smap_train'].flatten(), pred_train.flatten())[0, 1]
        corr_test = np.corrcoef(var_map['smap_test'].flatten(), pred_test.flatten())[0, 1]
    return corr_train, corr_test


#
# Printing & Plotting Definitions.
#
def print_confusion_stats():
    confusion_names = np.array([['TP', 'FP'], ['FN', 'TN']])
    confusion_stat_names = np.array([['TPR', 'FPR'], ['FNR', 'TNR']])
    confusion_pred_names = np.array([['PPV', 'FDR'], ['FOR', 'NPV']])
    confusion_stat_train, confusion_pred_train, confusion_acc_train = confusion_stats(matrix_train)
    confusion_stat_test, confusion_pred_test, confusion_acc_test = confusion_stats(matrix_test)
    print('Training Confusion Matrix:')
    print_matrix(confusion_names, matrix_train)
    print('Training Confusion Stats For True Condition:')
    print_matrix(confusion_stat_names, confusion_stat_train)
    print('Training Confusion Stats for Predicted Condition:')
    print_matrix(confusion_pred_names, confusion_pred_train)
    print('Training Confusion Accuracy:')
    print(confusion_acc_train)
    print()
    print('Testing Confusion Matrix:')
    print_matrix(confusion_names, matrix_test)
    print('Testing Confusion Stats For True Condition:')
    print_matrix(confusion_stat_names, confusion_stat_test)
    print('Testing Confusion Stats for Predicted Condition:')
    print_matrix(confusion_pred_names, confusion_pred_test)
    print('Testing Confusion Accuracy:')
    print(confusion_acc_test)
    print()


def print_matrix(names, matrix):
    for i in range(len(names)):
        print(f'{names[i]} = {matrix[i]}')
    print()


def print_correlations(corr_train, corr_test):
    print('Correlation Between SMAP Train and Pred Train:\nR = ', corr_train)
    print()
    print('Correlation Between SMAP Test and Pred Test:\nR = ', corr_test)
    print()


def print_to_file():
    original_stdout = sys.stdout
    # The 'a' denotes append. This opens the file in write mode and sets the stream to the end of the file.
    # The '+' opens the file in both read and write modes.
    with open(path_results + 'console_output.txt', 'a+') as output_log:
        sys.stdout = output_log
        output_log.seek(0)  # Sets the file pointer to the beginning of the file w/o affecting the write stream.
        if 'Correlation' not in output_log.read():
            if 'Fill_Value_Samples_Excluded' not in config.path_root:
                if 'masked' in config.loss:
                    pass
                else:
                    print_confusion_stats()
            print_correlations(correlation_train, correlation_test)
    sys.stdout = original_stdout


def scatter_plot(title, y_true, y_pred):
    fig = plt.figure()
    if 'Fill_Value_Samples_Excluded' not in config.path_root:
        plt.plot(y_true, y_pred, 'o', markersize=2)
    else:
        plt.plot(y_true.flatten(), y_pred.flatten(), 'o', markersize=2)
    plt.title(title + ' Set\nPredictions vs. Targets')
    plt.ylabel('Predictions')
    plt.xlim(plt.ylim())
    plt.xlabel('Targets')
    fig.savefig(path_results + 'scatter_' + title.lower() + '_plot.png')
    plt.pause(0.1)
    plt.close(fig)


def tol_plot(title, y_true, y_pred, bar_width):
    if 'Fill_Value_Samples_Excluded' not in config.path_root:
        error = np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))
    else:
        error = np.abs((y_true.flatten() - y_pred.flatten()) / y_true.flatten())
    tol = np.linspace(0, np.max(error), 100)
    error_below_tol = np.asarray([len(error[error <= tol[i]]) for i in range(len(tol))])
    dif = error_below_tol[1:] - error_below_tol[:-1]
    percent = error_below_tol / len(error) * 100
    fig, ax1 = plt.subplots()
    ax1.plot(tol * 100, percent, color='b')
    plt.title(title + ' Set\nNumber of Percent Error Elements Less Than A Given Tolerance')
    ax1.set_ylabel('Frequency [%]', color='b')
    ax1.set_xlabel('Tolerance [%]')
    plt.grid()
    ax2 = ax1.twinx()
    ax2.bar(tol[1:] * 100, dif, width=bar_width, color='r')
    ax2.set_ylabel('Difference in number of samples included in tolerance', color='r')
    plt.tight_layout()
    fig.savefig(path_results + 'tolerance_' + title.lower() + '_plot.png')
    plt.pause(0.1)
    plt.close(fig)


#
# Adding configuration capabilities to the file.
#
path_results = 'results/results_' + config_file_name.split('_')[-1] + '/'


#
# Load the model:
#
model = models.load_model(path_results + 'model_best.h5', custom_objects=func_map)


#
# Process results.
#
pred_train = model.predict(var_map[config.x_train])
pred_test = model.predict(var_map[config.x_test])
if 'Batch_2' in config.path_root:
    bar_width_train, bar_width_test = 4, 4
elif ('Batch_1' in config.path_root or '/home/ryan/Downloads/Dataset' in config.path_root and
      'Fill_Value_Samples_Excluded' in config.path_root):
    bar_width_train, bar_width_test = 0.4, 2
else:
    bar_width_train, bar_width_test = 2, 2
if 'Fill_Value_Samples_Excluded' not in config.path_root:
    if 'masked' in config.loss:
        mask_train = var_map['smap_train'] == smap_fill_value_scaled
        mask_test = var_map['smap_test'] == smap_fill_value_scaled
        scatter_plot('Training', var_map['smap_train'][~mask_train], pred_train[~mask_train])
        scatter_plot('Testing', var_map['smap_test'][~mask_test], pred_test[~mask_test])
        tol_plot('Training', var_map['smap_train'][~mask_train], pred_train[~mask_train], bar_width_train)
        tol_plot('Testing', var_map['smap_test'][~mask_test], pred_test[~mask_test], bar_width_test)
    else:
        matrix_train, tp_smap_train, tn_smap_train, fn_smap_train, fp_smap_train, \
            tp_pred_train, tn_pred_train, fn_pred_train, fp_pred_train = confusion(var_map['smap_train'], pred_train)
        matrix_test, tp_smap_test, tn_smap_test, fn_smap_test, fp_smap_test, \
            tp_pred_test, tn_pred_test, fn_pred_test, fp_pred_test = confusion(var_map['smap_test'], pred_test)
        scatter_plot('Training', tp_smap_train, tp_pred_train)
        scatter_plot('Testing', tp_smap_test, tp_pred_test)
        tol_plot('Training', tp_smap_train, tp_pred_train, bar_width_train)
        tol_plot('Testing', tp_smap_test, tp_pred_test, bar_width_test)
        print_confusion_stats()
else:
    scatter_plot('Training', var_map['smap_train'], pred_train)
    scatter_plot('Testing', var_map['smap_test'], pred_test)
    tol_plot('Training', var_map['smap_train'], pred_train, bar_width_train)
    tol_plot('Testing', var_map['smap_test'], pred_test, bar_width_test)

correlation_train, correlation_test = calculate_correlations()
print_correlations(correlation_train, correlation_test)

print_to_file()
print('Program Terminated')
