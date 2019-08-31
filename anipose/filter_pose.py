#!/usr/bin/env python3

from tqdm import tqdm, trange
import os.path, os
import numpy as np
import pandas as pd
from numpy import array as arr
from glob import glob
from scipy import signal
from scipy.interpolate import splev, splrep
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from collections import Counter

from .common import make_process_fun, natural_keys


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def sum_assign_thres(dists, thres=None):
    # prev_ind, new_ind = optimize.linear_sum_assignment(dists)
    from lapsolver import solve_dense
    prev_ind, new_ind = solve_dense(dists)
    if thres is not None:
        picked_dists = dists[prev_ind, new_ind]
        good = picked_dists < thres
        prev_ind = prev_ind[good]
        new_ind = new_ind[good]
    return prev_ind, new_ind

def assign_clusters(pts, max_offset=64, thres_dist=10):
    """takes in a set of points of shape NxPx2 where
    N: number of frames
    P: number of possible values
    returns an array of shape NxP of cluster ids
    """

    n_frames, n_possible, _ = pts.shape

    clusters = np.zeros((n_frames, n_possible), dtype=np.int64)
    clusters[:] = np.arange(clusters.size).reshape(clusters.shape)

    offsets = []
    offset = 1
    while offset < max_offset:
        offsets.append(offset)
        offset *= 2
    offsets.append(max_offset)

    for offset in offsets:
        step = max(int(offset / 2), 1)
        for i in range(0, n_frames-offset, step):
            kp_a = pts[i]
            kp_b = pts[i+offset]
            ix_a = np.where(~np.isnan(kp_a[:,0]))[0]
            ix_b = np.where(~np.isnan(kp_b[:,0]))[0]
            if len(ix_a) == 0 or len(ix_b) == 0:
                continue
            dists = cdist(kp_a[ix_a], kp_b[ix_b])
            prev_ind, new_ind = sum_assign_thres(dists, thres_dist)
            for prev, new in zip(ix_a[prev_ind], ix_b[new_ind]):
                cval = clusters[i+offset, new]
                nval = clusters[i, prev]
                clusters[clusters == cval] = nval

    return clusters


def remove_dups(pts, thres=4):
    tindex = np.repeat(np.arange(pts.shape[0])[:, None], pts.shape[1], axis=1)*100
    pts_ix = np.dstack([pts, tindex])
    tree = cKDTree(pts_ix.reshape(-1, 3))

    shape = (pts.shape[0], pts.shape[1])
    pairs = tree.query_pairs(thres)
    indices = [b for a, b in pairs]

    if len(pairs) == 0:
        return pts

    i0, i1 = np.unravel_index(indices, shape)
    pts_out = np.copy(pts)
    pts_out[i0, i1] = np.nan

    return pts_out

def find_best_path(points, scores, max_offset=64, thres_dist=20):
    """takes in a set of points of shape NxPx2 and an array of scores of shape NxP where
    N: number of frames
    P: number of possible values
    returns an array of shape Nx2 of picked points along
    with an array of length N of picked scores
    """

    points = remove_dups(points)
    clusters = assign_clusters(points, max_offset, thres_dist)

    score_clusters = np.zeros(clusters.shape)
    most_common = Counter(clusters.ravel()).most_common(n=20)
    for cnum, count in most_common:
        if count < 5: break
        check = clusters == cnum
        score = np.sum(scores[check])
        score_clusters[check] = score

    ixs_picked = np.argmax(score_clusters, axis=1)
    ixs = np.arange(len(points))
    points_picked = points[ixs, ixs_picked]
    scores_picked = scores[ixs, ixs_picked]

    return points_picked, scores_picked

def filter_pose_clusters(config, fname, outname):
    data_orig = pd.read_hdf(fname)
    scorer = data_orig.columns.levels[0][0]
    data = data_orig.loc[:, scorer]

    bp_index = data.columns.names.index('bodyparts')
    coord_index = data.columns.names.index('coords')
    bodyparts = list(data.columns.levels[bp_index])
    n_possible = len(data.columns.levels[coord_index])//3

    n_frames = len(data)
    n_joints = len(bodyparts)
    test = np.array(data).reshape(n_frames, n_joints, n_possible, 3)

    points_full = test[:, :, :, :2]
    scores_full = test[:, :, :, 2]

    points_full[scores_full < config['filter']['score_threshold']] = np.nan

    points = np.full((n_frames, n_joints, 2), np.nan, dtype='float64')
    scores = np.empty((n_frames, n_joints), dtype='float64')

    for jix in trange(n_joints, ncols=70):
        pts = points_full[:, jix, :]
        scs = scores_full[:, jix]
        pts_new, scs_new = find_best_path(pts, scs)
        points[:, jix] = pts_new
        scores[:, jix] = scs_new

    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

    dout = pd.DataFrame(columns=columns, index=data.index)

    dout.loc[:, (scorer, bodyparts, 'x')] = points[:, :, 0]
    dout.loc[:, (scorer, bodyparts, 'y')] = points[:, :, 1]
    dout.loc[:, (scorer, bodyparts, 'likelihood')] = scores

    dout.to_hdf(outname, 'df_with_missing', format='table', mode='w')
    

def filter_pose_medfilt(config, fname, outname):
    data_orig = pd.read_hdf(fname)
    scorer = data_orig.columns.levels[0][0]
    data = data_orig[scorer]

    bp_index = data.columns.names.index('bodyparts')
    bodyparts = list(data.columns.levels[bp_index])

    dout = data_orig.copy()

    for bp in bodyparts:

        x = arr(data[bp, 'x'])
        y = arr(data[bp, 'y'])
        score = arr(data[bp, 'likelihood'])
        # x, y, score = arr(data[bp]).T

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
    filter_type = config['filter']['type']

    assert filter_type in ['medfilt', 'clusters'], \
        "Invalid filter type, should be 'medfilt' or 'clusters', but found {}".format(filter_type)

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

        print(outpath)
        
        if config['filter']['type'] == 'medfilt':
            filter_pose_medfilt(config, fname, outpath)
        elif config['filter']['type'] == 'clusters':
            filter_pose_clusters(config, fname, outpath)


filter_pose_all = make_process_fun(process_session)
