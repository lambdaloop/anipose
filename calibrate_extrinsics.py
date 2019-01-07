#!/usr/bin/env python3

import cv2
from cv2 import aruco
from tqdm import tqdm, trange
import numpy as np
import sys
import itertools
import os, os.path
from glob import glob
from collections import defaultdict
import toml
from time import time
from pprint import pprint
sys.path.append('..')

from myconfig_pipeline import pipeline_prefix, pipeline_videos_raw, pipeline_calibration

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.GridBoard_create(2, 2, 4, 1, dictionary)


def detect_aruco(gray, intrinsics):
    # grayb = gray
    grayb = cv2.GaussianBlur(gray, (5, 5), 0)

    params = aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 100
    params.adaptiveThreshWinSizeMax = 600
    params.adaptiveThreshWinSizeStep = 50
    params.adaptiveThreshConstant = 5

    corners, ids, rejectedImgPoints = aruco.detectMarkers(grayb, dictionary,  parameters=params)

    INTRINSICS_K = np.array(intrinsics['camera_mat'])
    INTRINSICS_D = np.array(intrinsics['dist_coeff'])

    if ids is None:
        return [], []
    elif len(ids) < 2:
        return corners, ids

    detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
        aruco.refineDetectedMarkers(grayb, board, corners, ids,
                                    rejectedImgPoints,
                                    INTRINSICS_K, INTRINSICS_D,
                                    parameters=params)

    return detectedCorners, detectedIds

def estimate_pose(gray, intrinsics):

    detectedCorners, detectedIds = detect_aruco(gray, intrinsics)
    if len(detectedIds) < 3:
        return False, None

    INTRINSICS_K = np.array(intrinsics['camera_mat'])
    INTRINSICS_D = np.array(intrinsics['dist_coeff'])

    ret, rvec, tvec = aruco.estimatePoseBoard(detectedCorners, detectedIds, board,
                                              INTRINSICS_K, INTRINSICS_D)

    # flip the orientation as needed to make all the cameras align
    rotmat, _ = cv2.Rodrigues(rvec)
    test = np.dot([1,1,0], rotmat).dot([0,0,1])

    if test > 0:
        rvec[1,0] = -rvec[1,0]
        rvec[0,0] = -rvec[0,0]

    return True, (detectedCorners, detectedIds, rvec, tvec)


def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(rvec)
    out[:3,:3] = rotmat
    out[:3, 3] = tvec.flatten()
    out[3, 3] = 1
    return out


def mean_transform(M_list):
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in M_list]
    tvecs = [M[:3, 3] for M in M_list]

    rvec = np.median(rvecs, axis=0)
    tvec = np.median(tvecs, axis=0)

    return make_M(rvec, tvec)

def get_video_params(fname):
    cap = cv2.VideoCapture(fname)

    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return params

def get_folders(path):
    folders = next(os.walk(path))[1]
    return sorted(folders)

def get_cam_name(fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    return basename.split('_')[-1]

def get_video_name(fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    return '_'.join(basename.split('_')[0:-1])

def get_matrices(fname_dict, cam_intrinsics, skip=20):
    minlen = np.inf
    caps = dict()
    for cam_name, fname in fname_dict.items():
        cap = cv2.VideoCapture(fname)
        caps[cam_name] = cap
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        minlen = min(length, minlen)

    cam_names = fname_dict.keys()

    go = skip

    all_Ms = []

    for framenum in trange(minlen, ncols=70):
        M_dict = dict()

        for cam_name in cam_names:
            cap = caps[cam_name]
            ret, frame = cap.read()

            if framenum % skip != 0 and go <= 0:
                continue

            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            intrinsics = cam_intrinsics[cam_name]

            success, result = estimate_pose(gray, intrinsics)
            if not success:
                continue

            corners, ids, rvec, tvec = result

            M_dict[cam_name] = make_M(rvec, tvec)

        if len(M_dict) >= 2:
            go = skip
            all_Ms.append(M_dict)

        go = max(0, go-1)

    for cam_name, cap in caps.items():
        cap.release()

    return all_Ms


def get_transform(matrix_list, left, right):
    L = []
    for d in matrix_list:
        if left in d and right in d:
            M = np.matmul(d[left], np.linalg.inv(d[right]))
            L.append(M)
    M_mean = mean_transform(L)
    return M_mean


def get_all_matrix_pairs(matrix_list, cam_names):
    out = dict()
    
    for left in cam_names:
        for right in cam_names:
            if left == right:
                continue

            M = get_transform(matrix_list, left, right)
            out[(left, right)] = M

    return out

def get_extrinsics(fname_dicts, cam_intrinsics, skip=20):
    matrix_list = []
    cam_names = set()
    for fd in fname_dicts:
        ml = get_matrices(fd, cam_intrinsics, skip=skip)
        matrix_list.extend(ml)
        cam_names.update(fd.keys())
    
    pairs = get_all_matrix_pairs(matrix_list, sorted(cam_names))

    return pairs

def load_intrinsics(folder, cam_names):
    intrinsics = {}
    for cname in cam_names:
        fname = os.path.join(folder, 'intrinsics_{}.toml'.format(cname))
        intrinsics[cname] = toml.load(fname)
    return intrinsics
    

experiments = get_folders(pipeline_prefix)

for exp in experiments:
    exp_path = os.path.join(pipeline_prefix, exp)
    sessions = get_folders(exp_path)

    for session in sessions:
        print(session)

        videos = glob(os.path.join(pipeline_prefix, exp, session, pipeline_videos_raw, 'calib' + '*.avi'))
        videos = sorted(videos)

        cam_videos = defaultdict(list)

        cam_names = set()
        
        for vid in videos:
            name = get_video_name(vid)
            cam_videos[name].append(vid)
            cam_names.add(get_cam_name(vid))

        vid_names = cam_videos.keys()
        cam_names = sorted(cam_names)

        outname_base = 'extrinsics.toml'
        outdir = os.path.join(pipeline_prefix, exp, session, pipeline_calibration)
        os.makedirs(outdir, exist_ok=True)
        outname = os.path.join(outdir, outname_base)

        print(outname)
        if os.path.exists(outname):
            continue
        else:
            intrinsics = load_intrinsics(outdir, cam_names)
            
            fname_dicts = []
            for name in vid_names:
                fnames = cam_videos[name]
                cam_names = [get_cam_name(f) for f in fnames]
                fname_dict = dict(zip(cam_names, fnames))
                fname_dicts.append(fname_dict)
            
            extrinsics = get_extrinsics(fname_dicts, intrinsics)
            extrinsics_out = {}
            for k, v in extrinsics.items():
                new_key = k[0] + '_' + k[1]
                extrinsics_out[new_key] = v.tolist()

            with open(outname, 'w') as f:
                toml.dump(extrinsics_out, f)
