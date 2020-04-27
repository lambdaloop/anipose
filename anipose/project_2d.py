#!/usr/bin/env python3

import numpy as np
from glob import glob
import pandas as pd
import os.path
import cv2
from tqdm import tqdm, trange
from collections import defaultdict
from scipy import signal
import queue
import threading

from aniposelib.cameras import CameraGroup

from .common import make_process_fun, get_nframes, \
    get_video_name, get_cam_name, \
    get_video_params, get_video_params_cap, \
    get_data_length, natural_keys, true_basename, find_calibration_folder

from .triangulate import load_offsets_dict
from .filter_pose import write_pose_2d

def get_projected_points(config, pose_fname, cgroup, offsets_dict):

    pose_data = pd.read_csv(pose_fname)
    cols = [x for x in pose_data.columns if '_error' in x]
    bodyparts = [c.replace('_error', '') for c in cols]

    M = np.identity(3)
    center = np.zeros(3)
    for i in range(3):
        center[i] = np.mean(pose_data['center_{}'.format(i)])
        for j in range(3):
            M[i, j] = np.mean(pose_data['M_{}{}'.format(i, j)])

    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))

    all_points = np.array([np.array(pose_data.loc[:, (bp+'_x', bp+'_y', bp+'_z')])
                           for bp in bodyparts])

    all_errors = np.array([np.array(pose_data.loc[:, bp+'_error'])
                           for bp in bodyparts])

    all_scores = np.array([np.array(pose_data.loc[:, bp+'_score'])
                           for bp in bodyparts])
    
    if config['triangulation']['optim']:
        all_errors[np.isnan(all_errors)] = 0
    else:
        all_errors[np.isnan(all_errors)] = 10000
    good = (all_errors < 50)
    all_points[~good] = np.nan

    n_joints, n_frames, _ = all_points.shape
    n_cams = len(cgroup.cameras)

    all_points_flat = all_points.reshape(-1, 3)
    all_points_flat_t = (all_points_flat + center).dot(np.linalg.inv(M.T))

    points_2d_proj_flat = cgroup.project(all_points_flat_t)
    points_2d_proj = points_2d_proj_flat.reshape(n_cams, n_joints, n_frames, 2)

    cam_names = cgroup.get_names()
    for cix, cname in enumerate(cam_names):
        offset = offsets_dict[cname]
        dx, dy = offset[0], offset[1]
        points_2d_proj[cix, :, :, 0] -= dx
        points_2d_proj[cix, :, :, 1] -= dy

    return bodyparts, points_2d_proj, all_scores


def process_session(config, session_path):
    pipeline_videos_raw = config['pipeline']['videos_raw']
    pipeline_pose_3d = config['pipeline']['pose_3d']
    pipeline_pose_2d_projected = config['pipeline']['pose_2d_projected']

    video_ext = config['video_extension']

    vid_fnames_2d = glob(os.path.join(
        session_path, pipeline_videos_raw, "*."+video_ext))
    vid_fnames_2d = sorted(vid_fnames_2d, key=natural_keys)

    pose_fnames_3d = glob(os.path.join(
        session_path, pipeline_pose_3d, "*.csv"))
    pose_fnames_3d = sorted(pose_fnames_3d, key=natural_keys)
    
    if len(pose_fnames_3d) == 0:
        return

    fnames_2d = defaultdict(list)
    for vid in vid_fnames_2d:
        vidname = get_video_name(config, vid)
        fnames_2d[vidname].append(vid)

    fnames_3d = defaultdict(list)
    for fname in pose_fnames_3d:
        vidname = true_basename(fname)
        fnames_3d[vidname].append(fname)

    cgroup = None
    calib_folder = find_calibration_folder(config, session_path)
    if calib_folder is not None:
        calib_fname = os.path.join(calib_folder,
                                   config['pipeline']['calibration_results'],
                                   'calibration.toml')
        if os.path.exists(calib_fname):
            cgroup = CameraGroup.load(calib_fname)

    if cgroup is None:
        print('session {}: no calibration found, skipping'.format(session_path))
        return

    outdir = os.path.join(session_path, pipeline_pose_2d_projected)
    os.makedirs(outdir, exist_ok=True)

    for pose_fname in pose_fnames_3d:
        basename = true_basename(pose_fname)

        if len(fnames_2d[basename]) == 0:
            print(pose_fname, 'missing raw videos')
            continue

        fname_3d_current = pose_fname
        fnames_2d_current = fnames_2d[basename]
        fnames_2d_current = sorted(fnames_2d_current, key=natural_keys)

        out_fnames = [os.path.join(outdir, true_basename(fname) + '.h5')
                      for fname in fnames_2d_current]

        if all([os.path.exists(f) for f in out_fnames]):
            continue

        print(pose_fname)

        cam_names = [get_cam_name(config, fname)
                     for fname in fnames_2d_current]

        video_folder = os.path.join(session_path, pipeline_videos_raw)
        offsets_dict = load_offsets_dict(config, cam_names, video_folder)

        cgroup_subset = cgroup.subset_cameras_names(cam_names)

        bodyparts, points_2d_proj, all_scores = get_projected_points(
            config, fname_3d_current, cgroup_subset, offsets_dict)

        metadata = {
            'scorer': 'scorer',
            'bodyparts': bodyparts,
            'index': np.arange(points_2d_proj.shape[2])
        }

        n_cams, n_joints, n_frames, _ = points_2d_proj.shape
        
        pts = np.zeros((n_frames, n_joints, 3), dtype='float64')
        
        for cix, (cname, outname) in enumerate(zip(cam_names, out_fnames)):
            pts[:, :, :2] = points_2d_proj[cix].swapaxes(0, 1)
            pts[:, :, 2] = all_scores.T
            write_pose_2d(pts, metadata, outname)

project_2d_all = make_process_fun(process_session)
