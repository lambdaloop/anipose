#!/usr/bin/env python3

from sklearn.neural_network import MLPRegressor, MLPClassifier
import pandas as pd
import os.path
import numpy as np
from glob import glob
from ruamel.yaml import YAML
import pickle


def get_dataset_location(model_folder):
    config_fname = os.path.join(model_folder, 'config.yaml')

    yaml = YAML(typ='rt')
    with open(config_fname, 'r') as f:
        dlc_config = yaml.load(f)

    iternum = dlc_config['iteration']

    fname_pat = os.path.join(
        model_folder, 'training-datasets', 'iteration-'+str(iternum),
        '*', 'CollectedData_*.h5')
    fname = glob(fname_pat)[0]

    return fname

def load_pose_2d_training(fname):
    data_orig = pd.read_hdf(fname)
    scorer = data_orig.columns.levels[0][0]
    data = data_orig.loc[:, scorer]

    bp_index = data.columns.names.index('bodyparts')
    coord_index = data.columns.names.index('coords')
    bodyparts = list(data.columns.get_level_values(bp_index).unique())

    n_frames = len(data)
    n_joints = len(bodyparts)
    test = np.array(data).reshape(n_frames, n_joints, 2)

    bad = np.any(~np.isfinite(test), axis=2)
    test[bad] = np.nan

    metadata = {
        'bodyparts': bodyparts,
        'scorer': scorer,
        'index': data.index
    }

    return test, metadata

def generate_training_data(scores, n_iters=5):
    Xs = []
    ys = []

    for i in range(n_iters):
        scores_perturb = scores.copy()
        good = scores_perturb == 1
        scores_perturb[good] = np.random.normal(1, 0.3, size=np.sum(good))
        scores_perturb[~good] = np.random.normal(0, 0.3, size=np.sum(~good))
        flipped = np.random.uniform(size=good.shape) < 0.05
        scores_perturb = np.clip(scores_perturb, 0, 1)
        scores_perturb[flipped] = 1 - scores_perturb[flipped]
        Xs.append(scores_perturb)
        ys.append(scores)

    X = np.vstack(Xs)
    y = np.vstack(ys)

    return X, y

def train_mlp_classifier(X, y):
    hidden = int(X.shape[1] * 1.5)
    
    mlp = MLPClassifier(hidden_layer_sizes=(hidden),
                        verbose=2, max_iter=2000,
                        activation='tanh', tol=1e-5,
                        learning_rate='adaptive', solver='adam',
                        early_stopping=True)
    mlp.fit(X, y)

    return mlp


def save_mlp_classifier(mlp, fname):
    with open(fname, 'wb') as f:
        pickle.dump(mlp, f)
    print('autoencoder saved at:\n  {}'.format(fname))
 

def train_autoencoder(config):
    model_folder = config['model_folder']
    data_fname = get_dataset_location(model_folder)
    data, metadata = load_pose_2d_training(data_fname)
    n_frames, n_joints, _ = data.shape
    scores = np.ones((n_frames, n_joints), dtype='float64')
    bad = np.any(~np.isfinite(data), axis=2)
    scores[bad] = 0
    X, y = generate_training_data(scores)
    mlp = train_mlp_classifier(X, y)
    out_fname = os.path.join(config['path'], 'autoencoder.pickle')
    save_mlp_classifier(mlp, out_fname)

    
# model_folder = '/jellyfish/research/tuthill/hand-demo-dlc-TuthillLab-2019-08-05'

# config = {'model_folder': model_folder, 'path': model_folder}
# train_autoencoder(config)

# get dataset from deeplabcut folder
# generate augmented dataset to train autoencoder
# train MLP classifier
# save result
