#!/usr/bin/env python3

import cv2
# from cv2 import aruco
from tqdm import trange
import numpy as np
import os, os.path
from glob import glob
from collections import defaultdict
import pandas as pd

from .common import get_folders, true_basename, get_video_name
from .triangulate import load_offsets_dict, load_pose2d_fnames

from calligator.cameras import CameraGroup

def get_transform(row):
    M = np.identity(3)
    center = np.zeros(3)
    for i in range(3):
        center[i] = np.mean(row['center_{}'.format(i)])
        for j in range(3):
            M[i, j] = np.mean(row['M_{}{}'.format(i, j)])
    return M, center


def get_errors_group(config, group):
    pipeline_pose_3d = config['pipeline']['pose_3d']

    metadatas = dict()
    fnames_dict = dict()
    cam_names = []

    for cname, folder in group:
        metadata_fname = os.path.join('labeled-data', folder, 'anipose_metadata.csv')
        labels_fname = glob(os.path.join('labeled-data', folder, 'CollectedData*.h5'))[0]
        metadatas[cname] = pd.read_csv(metadata_fname)
        fnames_dict[cname] = labels_fname
        cam_names.append(cname)

    cam_names = sorted(cam_names)

    ## TODO: will have to modify this for custom offset per session
    offsets_dict = load_offsets_dict(config, cam_names)

    out = load_pose2d_fnames(fnames_dict, offsets_dict)

    points_labeled = out['points']
    bodyparts = out['bodyparts']

    metadata = metadatas[cam_names[0]]

    n_frames = len(metadata)
    n_joints = len(bodyparts)

    calib_fnames = np.array(metadata['calib'])
    points_3d_pred = np.full((n_frames, n_joints, 3), np.nan, 'float')
    points_3d_labeled = np.full((n_frames, n_joints, 3), np.nan, 'float')

    # get predicted 3D points
    paths_3d = []
    curr_path = None
    curr_pose = None
    curr_fnum = None
    for i in range(n_frames):
        row = metadata.iloc[i]
        fname = row['video']
        fnum = row['framenum']
        prefix = os.path.dirname(os.path.dirname(fname))
        vidname = get_video_name(config, fname)
        pose_path = os.path.join(prefix, pipeline_pose_3d, vidname + '.csv')
        paths_3d.append(pose_path)
        if curr_path != pose_path:
            curr_pose = pd.read_csv(pose_path)
            curr_fnum = np.array(curr_pose['fnum'])
            curr_path = pose_path
        try:
            ix = np.where(curr_fnum == fnum)[0][0]
        except IndexError:
            print("W: frame {} not found in 3D data for video {}".format(fnum, fname))
            continue
        row = curr_pose.iloc[ix]
        M, center = get_transform(row)
        pts = np.array([(row[bp+'_x'], row[bp+'_y'], row[bp+'_z']) for bp in bodyparts])
        pts_t = (pts + center).dot(np.linalg.inv(M.T))
        points_3d_pred[i] = pts_t

    # triangulate 3D points from labeled points
    curr_cgroup = None
    curr_calib_fname = None
    for i in range(n_frames):
        calib_fname = calib_fnames[i]
        if curr_calib_fname != calib_fname:
            curr_cgroup = CameraGroup.load(calib_fname)
            curr_calib_fname = calib_fname
        pts = points_labeled[:, i]
        points_3d_labeled[i] = curr_cgroup.triangulate(pts)

    errors = np.linalg.norm(points_3d_labeled - points_3d_pred, axis=2)

    out = pd.DataFrame()
    out['pose_path'] = paths_3d
    out['framenum'] = metadata['framenum']
    out['calib'] = metadata['calib']
    for bp_ix, bp in enumerate(bodyparts):
        # print(points_3d_labeled - points_3d_pred)
        out[bp + '_x_lab'] = points_3d_labeled[:, bp_ix, 0]
        out[bp + '_y_lab'] = points_3d_labeled[:, bp_ix, 1]
        out[bp + '_z_lab'] = points_3d_labeled[:, bp_ix, 2]
        out[bp + '_x_pred'] = points_3d_pred[:, bp_ix, 0]
        out[bp + '_y_pred'] = points_3d_pred[:, bp_ix, 1]
        out[bp + '_z_pred'] = points_3d_pred[:, bp_ix, 2]
        out[bp + '_error'] = errors[:, bp_ix]

    return out

def get_tracking_errors(config):
    # pipeline_videos_raw = config['pipeline']['videos_raw']
    group_folders = defaultdict(list)
    folders = get_folders('labeled-data')

    for folder in folders:
        group, _, cname = folder.rpartition('--')
        group_folders[group].append( (cname, folder) )

    datas = []
    for group, ffs in group_folders.items():
        print(group)
        dd = get_errors_group(config, ffs)
        datas.append(dd)
    data = pd.concat(datas)

    data.to_csv(os.path.join('summaries', 'tracking_errors.csv'),
                index=False)

    print('Errors saved in {}'.format(os.path.join('summaries', 'tracking_errors.py')))
