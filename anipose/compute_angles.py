#!/usr/bin/env python3

import numpy as np
from glob import glob
import pandas as pd
import os.path
from tqdm import tqdm, trange
import sys
from collections import defaultdict

from .common import make_process_fun, get_data_length, natural_keys


def compute_angles(config, labels_fname, outname):
    data = pd.read_csv(labels_fname)

    cols = [x for x in data.columns if '_error' in x]
    bodyparts = [c.replace('_error', '') for c in cols]

    vecs = dict()
    for bp in bodyparts:
        vec = np.array(data[[bp+'_x', bp+'_y', bp+'_z']])
        vecs[bp] = vec

    angle_names = config['angles'].keys()

    outdict = dict()
    outdict['fnum'] = data['fnum']

    for ang_name in angle_names:
        a,b,c = config['angles'][ang_name]

        v1 = vecs[a] - vecs[b]
        v2 = vecs[c] - vecs[b]

        v1 = v1 / np.linalg.norm(v1, axis=1)[:, None]
        v2 = v2 / np.linalg.norm(v2, axis=1)[:, None]

        ang_rad = np.arccos(np.sum(v1 * v2, axis=1))
        ang_deg = np.rad2deg(ang_rad)

        outdict[ang_name] = ang_deg

    dout = pd.DataFrame(outdict)
    dout.to_csv(outname, index=False)


def process_session(config, session_path):
    pipeline_3d = config['pipeline']['pose_3d']
    pipeline_angles = config['pipeline']['angles']

    labels_fnames = glob(os.path.join(session_path,
                                      pipeline_3d, '*.csv'))
    labels_fnames = sorted(labels_fnames, key=natural_keys)

    outdir = os.path.join(session_path, pipeline_angles)

    if len(labels_fnames) > 0:
        os.makedirs(outdir, exist_ok=True)

    for fname in labels_fnames:
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]

        out_fname = os.path.join(outdir, basename+'.csv')

        if os.path.exists(out_fname):
            continue

        print(out_fname)

        compute_angles(config, fname, out_fname)


compute_angles_all = make_process_fun(process_session)
