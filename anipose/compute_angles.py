#!/usr/bin/env python3

import numpy as np
from glob import glob
import pandas as pd
import os.path
from tqdm import tqdm, trange
import sys
from collections import defaultdict
from scipy.spatial.transform import Rotation

from .common import make_process_fun, get_data_length, natural_keys


# project v onto u
def proj(u, v):
    return u * (np.sum(v * u, axis = 1) / np.sum(u * u, axis = 1))[:,None]


# othogonalize u with respect to v
def ortho(u, v):
    return u - proj(v, u)


def normalize(u):
    return u / np.linalg.norm(u, axis = 1)[:, None]


def get_angles(vecs, angles):
    outdict = dict()
    angle_names = list(angles.keys())
    for ang_name in angle_names:
        angle = angles[ang_name]
        if angle[0] == "chain":
            d = angles_chain(vecs, angle[1:])
            for k, v in d.items():
                outdict[k] = v
        elif len(angle) == 4 and angle[0] == 'axis':
            outdict[ang_name] = angles_axis(vecs, angle[1:])
        elif len(angle) == 4 and angle[0] == 'cross-axis':
            outdict[ang_name] = angles_crossaxis(vecs, angle[1:])
        else: # 'flex'
            outdict[ang_name] = angles_flex(vecs, angle[-3:])
    return outdict


def angles_chain(vecs, chain_list):
    chain = []
    flex_type = []
    for c in chain_list:
        if c[-1] == "/":
            chain.append(c[:-1])
            flex_type.append(-1)
        else:
            chain.append(c)
            flex_type.append(1)

    n_joints = len(chain)
    keypoints = np.array([vecs[c] for c in chain])

    xfs = []
    cc = Rotation.identity()
    xfs.append(cc)

    for i in range(n_joints-1):
        pos = keypoints[i+1]
        z_dir = normalize(pos - keypoints[i])
        if i == n_joints - 2: # pick an arbitrary axis for the last joint
            x_dir = ortho([1, 0, 0], z_dir)
            if np.linalg.norm(x_dir) < 1e-5:
                x_dir = ortho([0, 1, 0], z_dir)
        else:
            x_dir = ortho(keypoints[i+2] - pos, z_dir)
            x_dir *= flex_type[i+1]
        x_dir = normalize(x_dir)
        y_dir = np.cross(z_dir, x_dir)
        M = np.dstack([x_dir, y_dir, z_dir])
        rot = Rotation.from_matrix(M)
        xfs.append(rot)

    angles = []
    for i in range(n_joints-1):
        rot = xfs[i].inv() * xfs[i+1]
        ang = rot.as_euler('zyx', degrees=True)
        if i != 0:
            flex = angles_flex(vecs, chain[i-1:i+2]) * flex_type[i]
            test = ~np.isclose(flex, ang[:,1])
            ang[:,0] += 180*test
            ang[:,1] = test*np.mod(-(ang[:,1]+180), 360) + (1-test)*ang[:,1]
            ang = np.mod(np.array(ang) + 180, 360) - 180
        angles.append(ang)

    outdict = dict()
    for i, (name, ang) in enumerate(zip(chain, angles)):
        outdict[name + "_flex"] = ang[:,1]
        if i != len(angles)-1:
            outdict[name + "_rot"] = ang[:,0]
        if i == 0:
            outdict[name + "_abduct"] = ang[:,2]

    return outdict


def angles_flex(vecs, angle):
    a,b,c = angle
    v1 = normalize(vecs[a] - vecs[b])
    v2 = normalize(vecs[c] - vecs[b])
    ang_rad = np.arccos(np.sum(v1 * v2, axis = 1))
    ang_deg = np.rad2deg(ang_rad)
    return ang_deg


def angles_axis(vecs, angle):    
    a,b,c = angle
    v1 = vecs[a] - vecs[b] 
    v2 = vecs[b] - vecs[c]
    z = normalize(v1)
    x = normalize(ortho([1, 0, 0], z))
    y = np.cross(z, x)
    ang_rad = np.arctan2(np.sum(v2 * y, axis = 1), np.sum(v2 * x, axis = 1))
    ang_deg = np.rad2deg(ang_rad)
    return ang_deg


def angles_crossaxis(vecs, angle):
    a,b,c = angle
    v1 = vecs[a] - vecs[b]
    v2 = vecs[b] -vecs[c]
    point = vecs[c] - vecs[a]
    z = normalize(np.cross(v1, v2))
    x = normalize(ortho([1, 0, 0], z))
    y = np.cross(z, x)
    ang_rad = np.arctan2(np.sum(point * y, axis = 1), np.sum(point * x, axis = 1))
    ang_deg = np.rad2deg(ang_rad)
    return ang_deg
    

def compute_angles(config, labels_fname, outname):
    data = pd.read_csv(labels_fname)

    cols = [x for x in data.columns if '_error' in x]
    bodyparts = [c.replace('_error', '') for c in cols]

    vecs = dict()
    for bp in bodyparts:
        vec = np.array(data[[bp+'_x', bp+'_y', bp+'_z']])
        vecs[bp] = vec
    
    outdict = get_angles(vecs, config.get('angles', dict()))
    outdict['fnum'] = data['fnum']
    
    dout = pd.DataFrame(outdict)
    dout.to_csv(outname, index=False)


def process_session(config, session_path):
    if 'angles' not in config: # don't process anything if no angles in config
        return
    
    if config['filter3d']['enabled']:
        pipeline_3d = config['pipeline']['pose_3d_filter']
    else:
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
