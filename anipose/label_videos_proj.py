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
from .project_2d import get_projected_points
from .label_videos import visualize_labels

## REFACTOR: this code is very similar to project_2d
def process_session(config, session_path):
    pipeline_videos_raw = config['pipeline']['videos_raw']
    pipeline_pose_3d = config['pipeline']['pose_3d']
    pipeline_videos_2d_projected = config['pipeline']['videos_2d_projected']

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

    outdir = os.path.join(session_path, pipeline_videos_2d_projected)
    os.makedirs(outdir, exist_ok=True)

    for pose_fname in pose_fnames_3d:
        basename = true_basename(pose_fname)

        if len(fnames_2d[basename]) == 0:
            print(pose_fname, 'missing raw videos')
            continue

        fname_3d_current = pose_fname
        fnames_2d_current = fnames_2d[basename]
        fnames_2d_current = sorted(fnames_2d_current, key=natural_keys)

        out_fnames = [os.path.join(outdir, true_basename(fname) + '.mp4')
                      for fname in fnames_2d_current]

        if all([os.path.exists(f) for f in out_fnames]):
            continue

        # print(pose_fname)

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
        
        for cix, (cname, vidname, outname) in enumerate(zip(cam_names, fnames_2d_current, out_fnames)):
            pts[:, :, :2] = points_2d_proj[cix].swapaxes(0, 1)
            pts[:, :, 2] = all_scores.T
            dlabs = write_pose_2d(pts, metadata, outname)

            if os.path.exists(outname) and \
               abs(get_nframes(outname) - get_nframes(vidname)) < 50:
                continue
            print(outname)
            visualize_labels(config, dlabs, vidname, outname)

label_proj_all = make_process_fun(process_session)
