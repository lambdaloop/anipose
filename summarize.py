#!/usr/bin/env python3

import numpy as np
from glob import glob
import pandas as pd
import os.path
from tqdm import tqdm, trange
import sys
from collections import defaultdict
from pprint import pprint

from common import process_all, true_basename, natural_keys

def get_angle_fnames(config, session_path):
    fnames = glob(os.path.join(session_path,
                               config['pipeline_angles'],
                               '*.csv'))
    return fnames

def get_pose3d_fnames(config, session_path):
    fnames = glob(os.path.join(session_path,
                               config['pipeline_pose_3d'],
                               '*.csv'))
    return fnames

def get_pose2d_fnames(config, session_path):
    fnames = glob(os.path.join(session_path,
                               config['pipeline_pose_2d'],
                               '*.h5'))
    return fnames

def make_summarize_fun(get_fnames_session, output_fname):
    
    def summarize_fun(config):
        output = process_all(config, get_fnames_session)

        datas = []
        items = sorted(output.items())

        for key,fnames in tqdm(items, ncols=70):
            fnames = sorted(fnames, key=natural_keys)
            for fname in fnames:
                d = pd.read_csv(fname)
                for num,foldername in enumerate(key, start=1):
                    k = 'folder_{}'.format(num)
                    d[k] = foldername

                d['filename'] = true_basename(fname)
                datas.append(d)

        dout = pd.concat(datas)
        dout['project'] = config['project']

        outdir = os.path.join(config['path'],
                              config['pipeline_summaries'])

        os.makedirs(outdir, exist_ok=True)

        outname = os.path.join(outdir, output_fname)

        print('Saving output...')
        dout.to_csv(outname)

    return summarize_fun

summarize_angles = make_summarize_fun(get_angle_fnames, 'angles.csv')
summarize_pose3d = make_summarize_fun(get_pose3d_fnames, 'pose_3d.csv')
