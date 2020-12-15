from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from src import utility
import gc
import sup_src.da_model as da_model

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import itertools
from matplotlib import rcParams


def get_class_weights(df_train):
    class_weights = []
    bla = compute_class_weight('balanced', range(len(df_train.iloc[:, -1].unique())), df_train.iloc[:, -1].values)
    for i, each in enumerate(bla):
        class_weights.append(each)
    return class_weights


def build_tissue_mlp(class_weights):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(.3))
    model.add(tf.keras.layers.Dense(len(class_weights), activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                  loss=loss_func_weighted(class_weights))

    return model

def build_sex_mlp(class_weights):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(len(class_weights), activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002435),
                  loss=loss_func_weighted(class_weights))
    return model

def loss_func_weighted(class_weights):
    def loss(y_true, y_pred):
        c_weights = tf.math.reduce_sum(tf.math.multiply(class_weights, y_true))
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return K.mean(loss * K.cast(c_weights, tf.float32))
    return loss

def mlp_report(target, model, classes, mapping):
    epoch_accuracy_target = tf.keras.metrics.CategoricalAccuracy()

    output = model(target[:, :-1])
    epoch_accuracy_target(output, tf.keras.utils.to_categorical(target[:, -1], classes))

    acc_sample = round(epoch_accuracy_target.result().numpy(), 4)

    res_mlp_model = mlp_report_per_class(target,
                                         model, classes, mapping)
    acc_tissue = round(res_mlp_model.groupby('data').mean().loc['target'].values[0],4)

    return acc_sample, acc_tissue

def mlp_report_per_class(target, model, classes, mapping):
    epoch_accuracy_target = tf.keras.metrics.CategoricalAccuracy()
    res = pd.DataFrame(columns=['tissue', 'accuracy', 'data'])
    for k, el in mapping.items():

        output = model(target[np.where(target[:, -1] == el), :-1][0])
        epoch_accuracy_target(output, tf.keras.utils.to_categorical(target[np.where(target[:, -1] == el), -1], classes))

        target_acc = epoch_accuracy_target.result()

        res_dic = {'tissue': k,
                   'accuracy': round(target_acc.numpy(), 4),
                   'data': 'target'}
        res = res.append(res_dic, ignore_index=True)

        epoch_accuracy_target.reset_states()

    return res


def class_to_int(source, target, bias=None, da_mode=False):
   
    dic_mapping = {}
    for i, each in enumerate(source.iloc[:,-1].unique()):
        dic_mapping[each] = i

    source.iloc[:,-1] = source.iloc[:,-1].apply(lambda x: dic_mapping[x])
    target.iloc[:,-1] = target.iloc[:,-1].apply(lambda x: dic_mapping[x])
    
    if da_mode:
        bias.iloc[:,-1] = bias.iloc[:,-1].apply(lambda x: dic_mapping[x])
        return source, target, bias, dic_mapping

    return source, target, dic_mapping


def run_experiment(source, 
                   target, 
                   bias=None,
                   mapping=None,
                   class_weights=None,
                   config=None,
                   model_type='mlp',
                   pheno_type='tissue',
                   name='test',
                   seeds=10,
                   mlp_epochs=10,
                   da_epochs=10,
                   batch_size=64,
                   save=False):
    
    '''
    Run a phenotype prediction experiment
    
    Parameters:
        source: training data, float32 array nxd, where n are the samples and d features d-1 are the integer encoded labels
        target: test data, array, see above
        mapping: dictionary mapping label strings to integers e.g. 'BRAIN': 0
        class_weights: class weights for loss function
        config: configuration dictionary for da model
        model_type: mlp or da
        pheno_type: tissue, sex or sample_source for correct MLP model selection
        name: name of experiment
        seeds: number of seeds to train
        mlp_epochs: number of epochs to train for an mlp or da in mlp mode
        da_epochs: number of epochs to train da in da mode
        batch_size: number of samples per batch
        save: if TRUE a model for each seed saved to models/ after epochs
        
    Return:
        Micro and macro accuracy on the test set for each seed as well as the average over all seeds
        
    '''
    classes = len(class_weights)
    results_sample = []
    results_tissue = []

    for i in range(seeds):
        if model_type == 'mlp':
            if pheno_type == 'tissue':
                model = build_tissue_mlp(class_weights)
            if pheno_type == "sample_source":
                model = build_tissue_mlp(class_weights)
            if pheno_type == "sex":
                model = build_sex_mlp(class_weights)

            model.fit(source[:, :-1], tf.keras.utils.to_categorical(source[:, -1]),
                      verbose=0,
                      epochs=mlp_epochs,
                      batch_size=batch_size)
            acc = mlp_report(target, model, classes, mapping)
            if save:
                model.save('models/' + name + '_' + str(i) + '.h5')
                
        elif model_type == 'da':
            model = da_model.DaModel(source, target, bias, mapping, config=config)
            model.train_source_mapper(epochs=mlp_epochs)
            model.train_bias_mapper(epochs=da_epochs)
            acc = mlp_report(target, model.target_model, classes, mapping)
            
            if save:
                model.target_model.save('models/' + name + '_' + str(i) + '.h5', overwrite=True)
        
        results_sample.append(acc[0])
        results_tissue.append(acc[1])


        del model
        gc.collect()

    print(results_sample)
    print(results_tissue)
    print(np.mean(results_sample))
    print(np.mean(results_tissue))


def load_and_predict(experiment, target, tissue_mapping, path, seeds=10, classes=16):
    mlp_gtex = []

    for i in range(seeds):
        mlp_gtex.append(tf.keras.models.load_model(path+experiment+'_'+ str(i) + '.h5', compile=False))

    res_mlp_gtex_sra = mlp_report_per_class(target, mlp_gtex[0], classes, tissue_mapping)
    acc_sample, acc_tissue = mlp_report(target, mlp_gtex[0], classes, tissue_mapping)
    acc_sample = [acc_sample]
    acc_tissue = [acc_tissue]

    for i in range(1, seeds):
        df_tmp = mlp_report_per_class(target, mlp_gtex[i], classes, tissue_mapping)
        res_mlp_gtex_sra = pd.concat([res_mlp_gtex_sra, df_tmp])

        acc_sample_tpm, acc_tissue_tmp = mlp_report(target, mlp_gtex[i], classes, tissue_mapping)
        acc_sample.append(acc_sample_tpm)
        acc_tissue.append(acc_tissue_tmp)

    res_mlp_gtex_sra['data'] = experiment

    return acc_sample, acc_tissue, res_mlp_gtex_sra

        
def label_sra_data(df_sra, df_meta, type_, with_zero=False):
    if with_zero:       
        df_type = df_meta
    else:
        df_type = df_meta[df_meta[type_] != '0']

    df_m = pd.merge(df_sra.drop(df_sra[~df_sra.index.isin(df_type.run_accession)].index),
                    df_type[['run_accession', type_]],
                    left_index=True, right_on='run_accession', how='left')
    df_m.set_index('run_accession', inplace=True)
    df_m[type_] = [x.upper() for x in df_m[type_]]

    return df_m