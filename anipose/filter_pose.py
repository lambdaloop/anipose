#!/usr/bin/env python3

from tqdm import tqdm, trange
import os.path, os
import numpy as np
import pandas as pd
from numpy import array as arr
from glob import glob
from scipy import signal
from scipy.interpolate import splev, splrep

from .common import make_process_fun, natural_keys


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def filter_pose(config, fname, outname):
    data_orig = pd.read_hdf(fname)
    scorer = data_orig.columns.levels[0][0]
    data = data_orig[scorer]

    bp_index = data.columns.names.index('bodyparts')
    bodyparts = list(data.columns.levels[bp_index])

    dout = data_orig.copy()

    for bp in bodyparts:

        x, y, score = arr(data[bp]).T

        xmed = signal.medfilt(x, kernel_size=config['filter']['medfilt'])
        ymed = signal.medfilt(y, kernel_size=config['filter']['medfilt'])

        errx = np.abs(x - xmed)
        erry = np.abs(y - ymed)
        err = errx + erry

        bad = np.zeros(len(x), dtype='bool')
        bad[err >= config['filter']['offset_threshold']] = True
        bad[score < config['filter']['score_threshold']] = True
        
        Xf = arr([x,y]).T
        Xf[bad] = np.nan

        Xfi = np.copy(Xf)

        for i in range(Xf.shape[1]):
            vals = Xfi[:, i]
            nans, ix = nan_helper(vals)
            # some data missing, but not too much
            if np.sum(nans) > 0 and np.mean(~nans) > 0.5 and np.sum(~nans) > 5:
                if config['filter']['spline']:
                    spline = splrep(ix(~nans), vals[~nans], k=3, s=0)
                    vals[nans]= splev(ix(nans), spline)
                else:
                    vals[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
            Xfi[:,i] = vals

        dout[scorer, bp, 'x'] = Xfi[:, 0]
        dout[scorer, bp, 'y'] = Xfi[:, 1]
        dout[scorer, bp, 'interpolated'] = np.isnan(Xf[:, 0])

    dout.to_hdf(outname, 'df_with_missing', format='table', mode='w')


def process_session(config, session_path):
    pipeline_pose = config['pipeline']['pose_2d']
    pipeline_pose_filter = config['pipeline']['pose_2d_filter']


    pose_folder = os.path.join(session_path, pipeline_pose)
    output_folder = os.path.join(session_path, pipeline_pose_filter)

    pose_files = glob(os.path.join(pose_folder, '*.h5'))
    pose_files = sorted(pose_files, key=natural_keys)

    if len(pose_files) > 0:
        os.makedirs(output_folder, exist_ok=True)

    for fname in pose_files:
        basename = os.path.basename(fname)
        outpath = os.path.join(session_path,
                               pipeline_pose_filter,
                               basename)

        if os.path.exists(outpath):
            continue

        print(fname)
        filter_pose(config, fname, outpath)


filter_pose_all = make_process_fun(process_session)
