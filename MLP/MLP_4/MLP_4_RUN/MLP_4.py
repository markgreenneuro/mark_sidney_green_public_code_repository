#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, mean_absolute_error
import sys
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from MLP_4_CONF_MAT_CLASS_REPORT_FORMAT import *
from MLP_4_OUTPUT_CONF_MAT_CLASS_REPORT import *
from MLP_4_OUTPUT_SUMMARY_STATS import *
from MLP_4_LOSS_ACCURACY_OUTPUT import *

import warnings


def get_params():
    ni_neurons = int(sys.argv[1])
    nj_neurons = int(sys.argv[2])
    nk_neurons = int(sys.argv[3])
    nl_neurons = int(sys.argv[4])
    epochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    optimizer = str(sys.argv[7])
    activation = str(sys.argv[8])
    return [ni_neurons, nj_neurons, nk_neurons, nl_neurons, epochs, batch_size, optimizer, activation]


def get_num_classes_default():
    num_classes = int(1)
    return num_classes


def set_data_root():
    data_dir = Path('/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE')
    return data_dir


def get_files(data_dir):
    os.chdir(data_dir)
    x_y_train = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised_train.csv')
    x_y_test = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised_test.csv')
    return [x_y_train, x_y_test]


def set_results_root():
    results_dir = Path('/home/markgreenneuroscience_gmail_com/RESULTS/MLP')
    if results_dir.exists():
        pass
    else:
        os.makedirs(results_dir)
    results_dir = str(results_dir)
    return results_dir


def get_make_stats_root(results_dir):
    stats_dir = Path.home().joinpath(results_dir, str('MLP_4_STATS'))
    if stats_dir.exists():
        pass
    else:
        os.makedirs(stats_dir)
    stats_dir = str(stats_dir)
    return stats_dir


def get_input_shape(x_y_train, x_y_test, num_classes):
    stutter_train = x_y_train['stutter']
    x_train = x_y_train.loc[:, x_y_train.columns != 'stutter']
    stutter_test = x_y_test['stutter']
    x_test = x_y_test.loc[:, x_y_test.columns != 'stutter']
    rows_length_train = x_train.shape[0]
    rows_length_test = x_test.shape[0]
    feature_vector_length = x_train.shape[1]
    input_shape = (feature_vector_length,)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.reshape(rows_length_train, feature_vector_length)
    x_test = x_test.reshape(rows_length_test, feature_vector_length)
    stutter_train = to_categorical(stutter_train, num_classes)
    stutter_test = to_categorical(stutter_test, num_classes)
    return [x_train, x_test, stutter_train, stutter_test, input_shape]


def build_mlp_4(x_train, y_train, ni_neurons, nj_neurons, nk_neurons, nl_neurons, epochs, batch_size, activation,
                optimizer, num_classes, input_shape):
    model = Sequential()
    model.add(Dense(ni_neurons, input_shape=input_shape))
    model.add(Activation(activation))
    model.add(Dense(nj_neurons))
    model.add(Activation(activation))
    model.add(Dense(nk_neurons))
    model.add(Activation(activation))
    model.add(Dense(nl_neurons))
    model.add(Activation(activation))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
    return model


def run_mlp_4(model, x_test, stutter_test, num_classes):
    test_results = model.evaluate(x_test, stutter_test, verbose=1)
    stutter_preds = np.argmax(model.predict(x_test), axis=-1)
    sys.stdout.write(str(stutter_preds))
    stutter_preds = to_categorical(stutter_preds, num_classes)
    class_report = classification_report(np.argmax(stutter_test, axis=1), np.argmax(stutter_preds, axis=1),
                                         zero_division=1, output_dict=True)
    class_report = pd.DataFrame(class_report).transpose()
    conf_mat = confusion_matrix(pd.DataFrame(stutter_test).values.argmax(axis=1),
                                pd.DataFrame(stutter_preds).values.argmax(axis=1))
    conf_mat = pd.DataFrame(conf_mat)
    test_results = pd.DataFrame([test_results])
    test_results.columns = ['Loss', 'Accuracy']
    return [test_results, stutter_preds, class_report, conf_mat]


def compute_roc_and_auc(stutter_test, stutter_preds, num_classes):
    # COMPUTE ROC CURVE AND AREA FOR EACH CLASSES
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0, num_classes):
        fpr[i], tpr[i], _ = roc_curve(stutter_preds[:, i], stutter_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # COMPUTE MICRO-AVERAGE ROC CURVE AND AREA
    fpr['micro'], tpr['micro'], _ = roc_curve(stutter_preds.ravel(), stutter_test.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    roc_auc = roc_auc['micro']
    roc_auc = pd.DataFrame([roc_auc])
    roc_auc.columns = ['ROC_AUC']
    return roc_auc


def get_num_params(model):
    num_params = model.count_params()
    return num_params


def calculate_aic(n, mse, num_params):
    if mse == 0:
        aic = 2 * num_params
    else:
        aic = n * np.log(mse) + 2 * num_params
    return aic


def calculate_bic(n, mse, num_params):
    if mse == 0:
        bic = num_params * np.log(n)
    else:
        bic = n * np.log(mse) + num_params * np.log(n)
    return bic


def get_mse_bic_aic(stutter_test, stutter_preds, model):
    num_params = get_num_params(model)
    mse = mean_absolute_error(stutter_test, stutter_preds)
    aic = calculate_aic(len(stutter_test), mse, num_params)
    bic = calculate_bic(len(stutter_test), mse, num_params)
    mse = pd.DataFrame([mse])
    mse.columns = ['MSE']
    bic = pd.DataFrame([bic])
    bic.columns = ['BIC']
    aic = pd.DataFrame([aic])
    aic.columns = ['AIC']
    return [mse, bic, aic]


def set_model_output(results_dir):
    model_dir = Path.home().joinpath(results_dir, 'MLP_4_MODELS')
    if model_dir.exists():
        pass
    else:
        os.makedirs(model_dir)
    model_dir = str(model_dir)
    return model_dir


def save_model(model_dir, model, name):
    model_name_path = str(model_dir) + '/' + str(name) + '.model'
    model.save(model_name_path)


def run_all_mlp_4() -> None:
    start_time = datetime.now()
    [ni_neurons, nj_neurons, nk_neurons, nl_neurons, epochs, batch_size, optimizer, activation] = get_params()
    num_classes = get_num_classes_default()
    data_dir = set_data_root()
    [x_y_train, x_y_test] = get_files(data_dir)
    results_dir = set_results_root()
    stats_dir = get_make_stats_root(results_dir)
    [x_train, x_test, stutter_train, stutter_test, input_shape] = get_input_shape(x_y_train, x_y_test, num_classes)
    model = build_mlp_4(x_train, stutter_train, ni_neurons, nj_neurons, nk_neurons, nl_neurons, epochs, batch_size,
                        activation, optimizer, num_classes, input_shape)
    [test_results, stutter_preds, class_report, conf_mat] = run_mlp_4(model, x_test, stutter_test, num_classes)
    [mse, bic, aic] = get_mse_bic_aic(stutter_test, stutter_preds, model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        roc_auc = compute_roc_and_auc(stutter_test, stutter_preds, num_classes)
    name = str(ni_neurons) + '_' + str(nj_neurons) + '_' + str(nk_neurons) + '_' + str(nl_neurons) + '_' + str(
        epochs) + '_' + str(batch_size) + '_' + str(activation) + '_' + str(batch_size) + '_' + str(
        optimizer) + '_' + str(activation)
    [class_report, conf_mat] = class_report_conf_mat(conf_mat, class_report, ni_neurons, nj_neurons, nk_neurons,
                                                     nl_neurons, epochs, batch_size,
                                                     optimizer, stutter_test)
    output_conf_mat_class_report(name, class_report, conf_mat, stats_dir)
    output_summary_stats(roc_auc, mse, bic, aic, ni_neurons, nj_neurons, nk_neurons, nl_neurons,
                         epochs, activation, batch_size, optimizer, num_classes, input_shape, name, stats_dir)
    format_and_output_loss_accuracy(test_results, ni_neurons, nj_neurons, nk_neurons, nl_neurons, epochs, batch_size,
                                    optimizer, stats_dir)
    model_dir = set_model_output(results_dir)
    save_model(model_dir, model, name)
    time_delta = datetime.now() - start_time
    exit_message = 'MLP 4 RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN ALL
run_all_mlp_4()
