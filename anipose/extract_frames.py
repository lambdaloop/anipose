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
from datetime import datetime
from ruamel.yaml import YAML
import shutil

from aniposelib.cameras import CameraGroup

from .common import make_process_fun, process_all, get_nframes, \
    get_video_name, get_cam_name, \
    get_video_params, get_video_params_cap, \
    get_data_length, natural_keys, true_basename, find_calibration_folder

from .triangulate import load_pose2d_fnames, load_offsets_dict

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def clean_index(df):
    df.index = [x.replace('\\', '/') for x in df.index]
    df.index = [x.replace('labeled-data/', '') for x in df.index]
    return df

def read_frames(caps_2d):
    frames_2d = []
    for cap in caps_2d:
        ret, frame = cap.read()
        if not ret:
            return False, None
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame
        frames_2d.append(img)

    return ret, frames_2d

def get_video_params_cap(cap):
    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['nframes'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)
    return params


def get_projected_points(bodyparts, pose_fname, cgroup, offsets_dict=None):
    pose_data = pd.read_csv(pose_fname)
    cols = [x for x in pose_data.columns if '_error' in x]


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

    n_joints, n_frames, _ = all_points.shape
    n_cams = len(cgroup.cameras)

    all_points_flat = all_points.reshape(-1, 3)
    all_points_flat_t = (all_points_flat + center).dot(np.linalg.inv(M.T))

    points_2d_proj_flat = cgroup.project(all_points_flat_t)
    points_2d_proj = points_2d_proj_flat.reshape(n_cams, n_joints, n_frames, 2)

    cam_names = cgroup.get_names()
    if offsets_dict is None:
        offsets_dict = dict([(cname, (0,0)) for cname in cam_names])

    for cix, cname in enumerate(cam_names):
        offset = offsets_dict[cname]
        dx, dy = offset[0], offset[1]
        points_2d_proj[cix, :, :, 0] -= dx
        points_2d_proj[cix, :, :, 1] -= dy

    return points_2d_proj.swapaxes(1, 2)

def get_pose2d_fnames(config, session_path):
    pipeline_pose = config['pipeline']['pose_2d']
    fnames = glob(os.path.join(session_path, pipeline_pose, '*.h5'))
    return session_path, fnames

def get_videos_fnames(config, session_path):
    pipeline_raw = config['pipeline']['videos_raw']
    ext = config['video_extension']
    fnames = glob(os.path.join(session_path, pipeline_raw, '*.' + ext))
    return session_path, fnames


def load_2d_data(config):
    pose_fnames = process_all(config, get_pose2d_fnames)
    cam_videos = defaultdict(list)

    for key, (session_path, fnames) in pose_fnames.items():
        for fname in fnames:
            # print(fname)
            vidname = get_video_name(config, fname)
            k = (key, session_path, vidname)
            cam_videos[k].append(fname)

    vid_names = sorted(cam_videos.keys())

    all_points = []
    all_scores = []
    all_proj = []
    all_fnames = []
    calib_fnames = []

    for name in tqdm(vid_names, desc='load points', ncols=80):
    # for name in vid_names:
        (key, session_path, vidname) = name
        fnames = sorted(cam_videos[name])
        cam_names = [get_cam_name(config, f) for f in fnames]
        fname_dict = dict(zip(cam_names, fnames))

        cgroup = None
        calib_folder = find_calibration_folder(config, session_path)
        if calib_folder is not None:
            calib_fname = os.path.join(calib_folder,
                                       config['pipeline']['calibration_results'],
                                       'calibration.toml')
            if os.path.exists(calib_fname):
                cgroup = CameraGroup.load(calib_fname)

        pose_fname = os.path.join(session_path, config['pipeline']['pose_3d'],
                                  vidname+'.csv')

        if cgroup is None or not os.path.exists(pose_fname):
            continue

        calib_fnames.append(calib_fname)

        video_folder = os.path.join(session_path, config['pipeline']['videos_raw'])
        offsets_dict = load_offsets_dict(config, cam_names, video_folder)
        out = load_pose2d_fnames(fname_dict)
        points_raw = out['points']
        scores = out['scores']
        bodyparts = out['bodyparts']

        vid_fnames = [os.path.join(session_path,
                                   config['pipeline']['videos_raw'],
                                   true_basename(f) + '.' + config['video_extension'])
                      for f in fnames]

        points_proj = get_projected_points(
            bodyparts, pose_fname, cgroup, offsets_dict)

        all_points.append(points_raw)
        all_scores.append(scores)
        all_proj.append(points_proj)
        all_fnames.append(vid_fnames)


    out = {
        'points': all_points,
        'scores': all_scores,
        'proj': all_proj,
        'fnames': all_fnames,
        'cam_names': cam_names,
        'calib_fnames': calib_fnames,
        'bodyparts': bodyparts
    }
    return out

def get_all_videos_fnames(config):
    vids_fnames = process_all(config, get_videos_fnames)
    cam_videos = defaultdict(list)

    for key, (session_path, fnames) in vids_fnames.items():
        for fname in fnames:
            # print(fname)
            vidname = get_video_name(config, fname)
            k = (key, session_path, vidname)
            cam_videos[k].append(fname)

    vid_names = sorted(cam_videos.keys(), key=lambda x: natural_keys(x[2]))

    all_fnames = []
    calib_fnames = []

    for name in tqdm(vid_names, desc='get videos', ncols=80):
        (key, session_path, vidname) = name
        fnames = sorted(cam_videos[name], key=natural_keys)
        cam_names = [get_cam_name(config, f) for f in fnames]

        cgroup = None
        calib_folder = find_calibration_folder(config, session_path)
        if calib_folder is not None:
            calib_fname = os.path.join(calib_folder,
                                       config['pipeline']['calibration_results'],
                                       'calibration.toml')
            if os.path.exists(calib_fname):
                cgroup = CameraGroup.load(calib_fname)

        # pose_fname = os.path.join(session_path, config['pipeline']['pose_3d'],
        #                           vidname+'.csv')

        if cgroup is None: # or not os.path.exists(pose_fname):
            continue

        calib_fnames.append(calib_fname)
        all_fnames.append(fnames)
        
    normal_count = max([len(x) for x in all_fnames])
    bad_num = np.sum([len(x) != normal_count for x in all_fnames])
    if bad_num > 0:
        print('W: ignored {} sets of videos with inconsistent number of cameras'.format(bad_num))
        all_fnames = [x for x in all_fnames if len(x) == normal_count]

    out = {
        'fnames': all_fnames,
        'calib_fnames': calib_fnames,
        'cam_names': cam_names
    }
    return out

def extract_frames_random(config, num_frames_pick=250):
    d = get_all_videos_fnames(config)
    all_fnames = d['fnames']
    cam_names = d['cam_names']
    calib_fnames = d['calib_fnames']

    main_folder = os.path.basename(os.getcwd())

    n_cams = len(cam_names)

    # model_folder = config['model_folder']
    # dlc_config_fname = os.path.join(model_folder, 'config.yaml')

    # yaml = YAML(typ='rt')
    # with open(dlc_config_fname, 'r') as f:
    #     dlc_config = yaml.load(f)

    vidnums = []
    framenums = []

    for vnum, fnames in enumerate(all_fnames):
        num_frames = np.inf
        for fname in fnames:
            params = get_video_params(fname)
            num_frames = min(num_frames, params['nframes'])
            
        vidnums.append(np.ones(num_frames, dtype='int64')*vnum)
        framenums.append(np.arange(num_frames))

    vidnums = np.hstack(vidnums)
    framenums = np.hstack(framenums)

    num_total = len(vidnums)
    
    indexes = np.arange(num_total)
    np.random.shuffle(indexes)
    
    check = np.ones(num_total, dtype='bool')

    picked = []
    for ix in indexes:
        vidnum = vidnums[ix]
        framenum = framenums[ix]
        if not check[ix]:
            continue

        similar = (vidnum == vidnums) & (np.abs(framenums - framenum) <= 10)
        check[similar] = False

        picked.append( (vidnum, framenum) )
        if len(picked) >= num_frames_pick:
            break

    picked = sorted(picked)

    nd = int(np.log10(num_frames_pick) + 1)
    img_format = 'img{:0' + str(nd) +'d}.png'
    images = [img_format.format(i) for i in range(num_frames_pick)]

    folder_base = '{}_{}_{}_random'.format(
        config['project'], main_folder, datetime.now().strftime('%Y-%m-%d_%H-%M'))

    folders = []
    metas = []
    indexes = []
    for cnum in range(n_cams):
        folder = folder_base + '--' + cam_names[cnum]
        images_cur = [os.path.join('labeled-data', folder_base, folder, img) for img in images]
        indexes.append(images_cur)

        meta = pd.DataFrame(columns=['img', 'calib', 'video', 'framenum'])
        metas.append(meta)

        # full_folder = os.path.join(model_folder, 'labeled-data', folder)
        full_folder = os.path.join('labeled-data', folder_base, folder)
        os.makedirs(full_folder, exist_ok=True)
        folders.append(full_folder)

    caps = []
    vidnum_cur = None

    max_width = defaultdict(int)
    max_height = defaultdict(int)

    rowcount = 0

    for imgnum, (vidnum, framenum) in enumerate(tqdm(picked, ncols=70)):
        # print(vidnum, framenum)
 
        if vidnum_cur != vidnum:
            for cap in caps:
                cap.release()
            caps = [cv2.VideoCapture(f) for f in all_fnames[vidnum]]
            vidnum_cur = vidnum
            framenum_cur = 0

        while framenum_cur < framenum:
            ret, frames = read_frames(caps)
            framenum_cur += 1
            if not ret:
                break

        ret, frames = read_frames(caps)
        framenum_cur += 1
        if not ret:
            continue

        for cnum in range(n_cams):
            row = indexes[cnum][imgnum]

            metas[cnum].loc[rowcount, 'img'] = row
            metas[cnum].loc[rowcount, 'calib'] = calib_fnames[vidnum]
            metas[cnum].loc[rowcount, 'video'] = all_fnames[vidnum][cnum]
            metas[cnum].loc[rowcount, 'framenum'] = framenum

            cv2.imwrite(row, frames[cnum])
            height, width, _  = frames[cnum].shape
            max_height[cnum] = max(max_height[cnum], height)
            max_width[cnum] = max(max_width[cnum], width)

        rowcount += 1

    for cnum, folder in enumerate(folders):
        metas[cnum].to_csv(os.path.join(folder, 'anipose_metadata.csv'), index=False)

        # key = true_basename(folder) + '.avi'
        # dlc_config['video_sets'][key] = {
        #     'crop': '0, {}, 0, {}'.format(max_height[cnum], max_width[cnum])
        # }

    config_path = os.path.join(config['path'], 'config.toml')
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join('labeled-data', folder_base, 'config.toml'))

    calib_path = os.path.join(config['path'],
                              config['pipeline']['calibration_results'],
                              'calibration.toml')
    if os.path.exists(calib_path):
        shutil.copy(calib_path, os.path.join('labeled-data', folder_base, 'calibration.toml'))

    # with open(dlc_config_fname, 'w') as f:
    #     yaml.dump(dlc_config, f)

    
POSSIBLE_MODES = ['good', 'bad', 'random']
def extract_frames_picked(config, mode='bad', num_frames_pick=250, scorer=None):
    if mode not in POSSIBLE_MODES:
        raise ValueError(
            'extract_frames_picked needs mode to be one of {}, but received "{}"'
            .format(POSSIBLE_MODES, mode))

    print('loading tracked data...')
    d = load_2d_data(config)

    print('getting {} frames...'.format(mode))
    points = d['points']
    proj = d['proj']
    scores = d['scores']
    fnames = d['fnames']
    cam_names = d['cam_names']
    calib_fnames = d['calib_fnames']
    bodyparts = d['bodyparts']

    main_folder = os.path.basename(os.getcwd())

    model_folder = config['model_folder']
    dlc_config_fname = os.path.join(model_folder, 'config.yaml')


    if scorer is None:
        # try to read it from yaml file
        if os.path.exists(dlc_config_fname):
            yaml = YAML(typ='rt')
            with open(dlc_config_fname, 'r') as f:
                dlc_config = yaml.load(f)
            scorer = dlc_config['scorer']
        else:
            scorer = 'default'

    nums = [p.shape[1] for p in points]
    num_total = np.sum(nums)
    n_cams = proj[0].shape[0]

    vidnums = np.zeros(num_total, dtype='int64')
    framenums = np.zeros(num_total, dtype='int64')
    errors = np.zeros(num_total, dtype='float64')

    start = 0
    for vnum, num_frames in enumerate(nums):
        a = start
        b = start + num_frames

        errors_cur = np.linalg.norm(proj[vnum] - points[vnum], axis=3)
        errors_mean = np.mean(np.mean(errors_cur, axis=2), axis=0)

        vidnums[a:b] = vnum
        framenums[a:b] = np.arange(num_frames)
        errors[a:b] = errors_mean
        start += num_frames

    good = np.isfinite(errors)
    errors[~good] = np.max(errors[good])*0.5

    if mode == 'bad':
        log_errors = np.log(errors+0.1)
        log_errors = np.clip(log_errors, -np.inf, np.percentile(log_errors, 85))
        error_percent = np.max(log_errors) - np.percentile(log_errors, 60)
        noise = np.random.uniform(0, error_percent, size=errors.shape)
        indexes = np.argsort(-log_errors + noise)
    elif mode == 'good':
        error_percent = np.percentile(errors, 20)
        noise = np.random.uniform(-error_percent, error_percent, size=errors.shape)
        indexes = np.argsort(errors + noise)
    elif mode == 'random':
        indexes = np.arange(len(errors))
        np.random.shuffle(indexes)

    check = np.ones(num_total, dtype='bool')

    picked = []
    for ix in indexes:
        vidnum = vidnums[ix]
        framenum = framenums[ix]
        if not check[ix]:
            continue

        similar = (vidnum == vidnums) & (np.abs(framenums - framenum) <= 10)
        check[similar] = False

        picked.append( (vidnum, framenum) )
        if len(picked) >= num_frames_pick:
            break

    picked = sorted(picked)


    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ['x', 'y']],
        names=['scorer', 'bodyparts', 'coords'])

    nd = int(np.log10(num_frames_pick) + 1)
    img_format = 'img{:0' + str(nd) +'d}.png'
    images = [img_format.format(i) for i in range(num_frames_pick)]

    # folder_base = datetime.now().strftime('%Y-%m-%d--%H-%M')
    folder_base = '{}_{}_{}_{}'.format(
        config['project'], main_folder, datetime.now().strftime('%Y-%m-%d_%H-%M'), mode)

    folders = []
    douts = []
    metas = []
    indexes = []
    for cnum in range(n_cams):
        folder = folder_base + '--' + cam_names[cnum]
        images_cur = [os.path.join('labeled-data', folder_base, folder, img) for img in images]
        indexes.append(images_cur)

        dout = pd.DataFrame(columns=columns)
        dout = dout.astype('float64')
        douts.append(dout)

        meta = pd.DataFrame(columns=['img', 'calib', 'video', 'framenum'])
        metas.append(meta)

        # full_folder = os.path.join(model_folder, 'labeled-data', folder)
        full_folder = os.path.join('labeled-data', folder_base, folder)
        os.makedirs(full_folder, exist_ok=True)
        folders.append(full_folder)

    caps = []
    vidnum_cur = None

    max_width = defaultdict(int)
    max_height = defaultdict(int)

    rowcount = 0

    for imgnum, (vidnum, framenum) in enumerate(tqdm(picked, ncols=70)):
        # print(vidnum, framenum)

        if vidnum_cur != vidnum:
            for cap in caps:
                cap.release()
            caps = [cv2.VideoCapture(f) for f in fnames[vidnum]]
            vidnum_cur = vidnum
            framenum_cur = 0

        while framenum_cur < framenum:
            ret, frames = read_frames(caps)
            framenum_cur += 1
            if not ret:
                break

        ret, frames = read_frames(caps)
        framenum_cur += 1
        if not ret:
            continue

        pred = proj[vidnum][:, framenum]
        sc = scores[vidnum][:, framenum]
        pred[sc < 0.3] = np.nan

        for cnum in range(n_cams):
            row = indexes[cnum][imgnum]
            row_x = row.replace(folder_base + '/', '')
            douts[cnum].loc[row_x, (scorer, bodyparts, 'x')] = pred[cnum, :, 0]
            douts[cnum].loc[row_x, (scorer, bodyparts, 'y')] = pred[cnum, :, 1]

            metas[cnum].loc[rowcount, 'img'] = row
            metas[cnum].loc[rowcount, 'calib'] = calib_fnames[vidnum]
            metas[cnum].loc[rowcount, 'video'] = fnames[vidnum][cnum]
            metas[cnum].loc[rowcount, 'framenum'] = framenum

            cv2.imwrite(row, frames[cnum])
            height, width, _  = frames[cnum].shape
            max_height[cnum] = max(max_height[cnum], height)
            max_width[cnum] = max(max_width[cnum], width)

        rowcount += 1

    for cnum, folder in enumerate(folders):
        douts[cnum].to_hdf(os.path.join(folder, 'CollectedData_' + scorer + '.h5'),
                      key='df_with_missing', format='table', mode='w')
        douts[cnum].to_csv(os.path.join(folder, 'CollectedData_' + scorer + '.csv'))

        metas[cnum].to_csv(os.path.join(folder, 'anipose_metadata.csv'), index=False)

        key = true_basename(folder) + '.avi'
        # dlc_config['video_sets'][key] = {
        #     'crop': '0, {}, 0, {}'.format(max_height[cnum], max_width[cnum])
        # }

    data_concat = pd.concat([clean_index(d.copy()) for d in douts])
    data_concat.to_csv(os.path.join('labeled-data', folder_base, 'CollectedData.csv'), index=True)

    config_path = os.path.join(config['path'], 'config.toml')
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join('labeled-data', folder_base, 'config.toml'))

    calib_path = os.path.join(config['path'],
                              config['pipeline']['calibration_results'],
                              'calibration.toml')
    if os.path.exists(calib_path):
        shutil.copy(calib_path, os.path.join('labeled-data', folder_base, 'calibration.toml'))

    # calibration_path =
    # with open(dlc_config_fname, 'w') as f:
    #     yaml.dump(dlc_config, f)
