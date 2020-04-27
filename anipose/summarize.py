#!/usr/bin/env python3

import numpy as np
from glob import glob
import pandas as pd
import os.path
from tqdm import tqdm, trange
import sys
from collections import defaultdict
from pprint import pprint

from .common import process_all, true_basename, natural_keys, get_cam_name

def get_angle_fnames(config, session_path):
    fnames = glob(os.path.join(session_path,
                               config['pipeline']['angles'],
                               '*.csv'))
    return fnames

def get_pose3d_fnames(config, session_path):
    fnames = glob(os.path.join(session_path,
                               config['pipeline']['pose_3d'],
                               '*.csv'))
    return fnames

def get_pose3d_filtered_fnames(config, session_path):
    fnames = glob(os.path.join(session_path,
                               config['pipeline']['pose_3d_filter'],
                               '*.csv'))
    return fnames

def get_pose2d_fnames(config, session_path):
    fnames = glob(os.path.join(session_path,
                               config['pipeline']['pose_2d'],
                               '*.h5'))
    return fnames

def get_pose2d_filtered_fnames(config, session_path):
    fnames = glob(os.path.join(session_path,
                               config['pipeline']['pose_2d_filter'],
                               '*.h5'))
    return fnames

def make_summarize_fun(get_fnames_session, output_fname, h5=False):

    def summarize_fun(config):
        output = process_all(config, get_fnames_session)

        datas = []
        items = sorted(output.items())

        for key,fnames in tqdm(items, ncols=70, desc='sessions'):
            fnames = sorted(fnames, key=natural_keys)
            for fname in tqdm(fnames, ncols=70, desc='files'):
                if h5:
                    d = pd.read_hdf(fname)
                    scorer = d.columns.levels[0][0]
                    d = d[scorer]
                else:
                    d = pd.read_csv(fname)

                for num,foldername in enumerate(key, start=1):
                    k = 'folder_{}'.format(num)
                    d[k] = foldername

                d['filename'] = true_basename(fname)
                datas.append(d)

        dout = pd.concat(datas)
        dout['project'] = config['project']

        outdir = os.path.join(config['path'],
                              config['pipeline']['summaries'])

        os.makedirs(outdir, exist_ok=True)

        outname = os.path.join(outdir, output_fname)

        print('Saving output...')
        dout.to_csv(outname, index=False)

        if h5:
            basename = true_basename(outname)
            outfull_new = os.path.join(outdir, basename+'.h5')
            dout.to_hdf(outfull_new, 'df_with_missing', format='table', mode='w')

    return summarize_fun


summarize_angles = make_summarize_fun(get_angle_fnames, 'angles.csv', h5=False)
summarize_pose3d = make_summarize_fun(get_pose3d_fnames, 'pose_3d.csv', h5=False)
summarize_pose3d_filtered = make_summarize_fun(get_pose3d_fnames, 'pose_3d_filtered.csv', h5=False)

summarize_pose2d = make_summarize_fun(get_pose2d_fnames, 'pose_2d.csv', h5=True)
summarize_pose2d_filtered = make_summarize_fun(get_pose2d_filtered_fnames, 'pose_2d_filtered.csv', h5=True)

def summarize_errors(config):
    output = process_all(config, get_pose2d_filtered_fnames)

    rows = []
    items = sorted(output.items())

    for key,fnames in tqdm(items, ncols=70, desc='sessions'):
        fnames = sorted(fnames, key=natural_keys)
        for fname in tqdm(fnames, ncols=70, desc='files'):
            data = pd.read_hdf(fname)
            scorer = data.columns.levels[0][0]
            data = data[scorer]

            bp_index = data.columns.names.index('bodyparts')
            bodyparts = list(data.columns.levels[bp_index])

            rates_row = dict()
            for bp in bodyparts:
                rates_row[bp] = np.mean(data[bp]['interpolated'])

            for num,foldername in enumerate(key, start=1):
                k = 'folder_{}'.format(num)
                rates_row[k] = foldername

            rates_row['filename'] = true_basename(fname)
            rates_row['cam_name'] = get_cam_name(config, fname)
            rows.append(rates_row)

    dout = pd.DataFrame(rows)
    dout['project'] = config['project']

    outdir = os.path.join(config['path'],
                          config['pipeline']['summaries'])

    os.makedirs(outdir, exist_ok=True)

    outname = os.path.join(outdir, 'errors.csv')

    print('Saving output...')
    dout.to_csv(outname, index=False)
