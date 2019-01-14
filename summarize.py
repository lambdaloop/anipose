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

def process_angles_session(config, session_path):
    pipeline_angles = config['pipeline_angles']

    fnames = glob(os.path.join(session_path,
                               pipeline_angles, '*.csv'))

    return fnames


def summarize_angles(config):
    # TODO: fill this is in using process_all
    output = process_all(config, process_angles_session)

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

    print('Concatentating outputs...')
    dout = pd.concat(datas)
    dout['project'] = config['project']

    outdir = os.path.join(config['path'],
                          config['pipeline_summaries'])

    os.makedirs(outdir, exist_ok=True)

    outname = os.path.join(outdir, 'angles.csv')

    print('Saving output...')
    dout.to_csv(outname)
