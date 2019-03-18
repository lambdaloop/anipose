#!/usr/bin/env python3

import cv2
from cv2 import aruco
from tqdm import trange
import numpy as np
import os, os.path
from glob import glob
from collections import defaultdict
import toml

from .common import \
    find_calibration_folder, make_process_fun, \
    get_cam_name, get_video_name, load_intrinsics, \
    get_calibration_board

def fill_points(corners, ids):
    # TODO: this should change with calibration board config
    # 16 comes from 4 boxes (2x2) with 4 corners each
    out = np.zeros((16, 2))
    out.fill(np.nan)

    if ids is None:
        return out

    for id_wrap, corner_wrap in zip(ids, corners):
        ix = id_wrap[0]
        corner = corner_wrap.flatten().reshape(4,2)
        if ix >= 4: continue
        out[ix*4:(ix+1)*4,:] = corner

    return out

def detect_aruco(gray, intrinsics, board):
    # grayb = gray
    grayb = cv2.GaussianBlur(gray, (5, 5), 0)

    params = aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 100
    params.adaptiveThreshWinSizeMax = 600
    params.adaptiveThreshWinSizeStep = 50
    params.adaptiveThreshConstant = 5

    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        grayb, board.dictionary, parameters=params)

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

def estimate_pose(gray, intrinsics, board):

    detectedCorners, detectedIds = detect_aruco(gray, intrinsics, board)
    if len(detectedIds) < 3:
        return False, None

    INTRINSICS_K = np.array(intrinsics['camera_mat'])
    INTRINSICS_D = np.array(intrinsics['dist_coeff'])

    ret, rvec, tvec = aruco.estimatePoseBoard(detectedCorners, detectedIds, board,
                                              INTRINSICS_K, INTRINSICS_D)

    # flip the orientation as needed to make all the cameras align
    rotmat, _ = cv2.Rodrigues(rvec)
    # test = np.dot([1,1,0], rotmat).dot([0,0,1])
    # if test > 0:
    #     rvec[1,0] = -rvec[1,0]
    #     rvec[0,0] = -rvec[0,0]

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

    rvec = np.mean(rvecs, axis=0)
    tvec = np.mean(tvecs, axis=0)

    return make_M(rvec, tvec)

def mean_transform_robust(M_list, approx=None, error=0.3):
    if approx is None:
        M_list_robust = M_list
    else:
        M_list_robust = []
        for M in M_list:
            rot_error = (M - approx)[:3,:3]
            m = np.max(np.abs(rot_error))
            if m < error:
                M_list_robust.append(M)
    return mean_transform(M_list_robust)

def get_matrices(fname_dict, cam_intrinsics, board, skip=20):
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

            success, result = estimate_pose(gray, intrinsics, board)
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
    M_mean = mean_transform_robust(L, M_mean, error=0.5)
    M_mean = mean_transform_robust(L, M_mean, error=0.2)
    M_mean = mean_transform_robust(L, M_mean, error=0.1)
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

def get_extrinsics(fname_dicts, cam_intrinsics, board, skip=20):
    ## TODO optimize transforms based on reprojection errors
    matrix_list = []
    cam_names = set()
    for fd in fname_dicts:
        ml = get_matrices(fd, cam_intrinsics, board, skip=skip)
        matrix_list.extend(ml)
        cam_names.update(fd.keys())

    pairs = get_all_matrix_pairs(matrix_list, sorted(cam_names))

    return pairs

def process_session(config, session_path):
    # pipeline_videos_raw = config['pipeline']['videos_raw']
    pipeline_calibration_videos = config['pipeline']['calibration_videos']
    pipeline_calibration_results = config['pipeline']['calibration_results']

    calibration_path = find_calibration_folder(config, session_path)

    if calibration_path is None:
        return

    videos = glob(os.path.join(calibration_path,
                               pipeline_calibration_videos,
                               '*.avi'))
    videos = sorted(videos)

    cam_videos = defaultdict(list)

    cam_names = set()

    for vid in videos:
        name = get_video_name(config, vid)
        cam_videos[name].append(vid)
        cam_names.add(get_cam_name(config, vid))

    vid_names = cam_videos.keys()
    cam_names = sorted(cam_names)

    outname_base = 'extrinsics.toml'
    outdir = os.path.join(calibration_path, pipeline_calibration_results)
    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, outname_base)

    board = get_calibration_board(config)

    print(outname)
    if os.path.exists(outname):
        return
    else:
        intrinsics = load_intrinsics(outdir, cam_names)

        fname_dicts = []
        for name in vid_names:
            fnames = cam_videos[name]
            cam_names = [get_cam_name(config, f) for f in fnames]
            fname_dict = dict(zip(cam_names, fnames))
            fname_dicts.append(fname_dict)

        extrinsics = get_extrinsics(fname_dicts, intrinsics, board)
        extrinsics_out = {}
        for k, v in extrinsics.items():
            new_key = k[0] + '_' + k[1]
            extrinsics_out[new_key] = v.tolist()

        with open(outname, 'w') as f:
            toml.dump(extrinsics_out, f)


calibrate_extrinsics_all = make_process_fun(process_session)
