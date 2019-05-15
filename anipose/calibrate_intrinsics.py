#!/usr/bin/env python3

import cv2
from cv2 import aruco
from tqdm import trange
import numpy as np
import itertools
import os, os.path
from glob import glob
from collections import defaultdict
import toml
from time import time

from checkerboard import detect_checkerboard

from .common import make_process_fun, get_cam_name, find_calibration_folder, \
    get_video_params, get_calibration_board, get_board_type, get_board_size, get_expected_corners


def get_corners_aruco(fname, board, skip=20):
    cap = cv2.VideoCapture(fname)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    allCorners = []
    allIds = []

    go = int(skip/2)

    board_type = get_board_type(board)
    board_size = get_board_size(board)

    max_size = get_expected_corners(board)

    for framenum in trange(length, ncols=70):
        ret, frame = cap.read()
        if not ret:
            break

        if framenum % skip != 0 and go <= 0:
            continue

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        params = aruco.DetectorParameters_create()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshWinSizeMin = 100
        params.adaptiveThreshWinSizeMax = 700
        params.adaptiveThreshWinSizeStep = 50
        params.adaptiveThreshConstant = 5

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, board.dictionary, parameters=params)

        if corners is None or len(corners) <= 2:
            go = max(0, go-1)
            continue

        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
            aruco.refineDetectedMarkers(gray, board, corners, ids,
                                        rejectedImgPoints, parameters=params)

        if board_type == 'charuco' and len(detectedCorners) > 0:
            ret, detectedCorners, detectedIds = aruco.interpolateCornersCharuco(
                detectedCorners, detectedIds, gray, board)

        if detectedCorners is not None and \
           len(detectedCorners) >= 2 and len(detectedCorners) <= max_size:
            allCorners.append(detectedCorners)
            allIds.append(detectedIds)
            go = int(skip/2)

        go = max(0, go-1)

    cap.release()

    return allCorners, allIds


def get_corners_checkerboard(fname, board, skip=20):
    cap = cv2.VideoCapture(fname)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    allCorners = []
    allScores = []

    go = int(skip/2)

    board_size = board.getChessboardSize()

    for framenum in trange(length, ncols=70):
        ret, frame = cap.read()
        if not ret:
            break

        if framenum % skip != 0 and go <= 0:
            continue

        grayf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ## TODO: adjust the checkerboard library to handle more general ratios
        ## TODO: make this ratio and trimming configurable
        ratio = 400.0/grayf.shape[0]
        gray = cv2.resize(grayf, (0,0), fx=ratio, fy=ratio,
                          interpolation=cv2.INTER_CUBIC)

        corners, check_score = detect_checkerboard(gray, board_size)

        if corners is not None and len(corners) > 0:
            corners_new = corners / ratio
            allCorners.append(corners_new)
            allScores.append(check_score)
            go = int(skip/2)

        go = max(0, go-1)

    cap.release()

    return allCorners, allScores


def trim_corners(allCorners, allIds, maxBoards=85):
    counts = np.array([len(cs) for cs in allCorners])
    sort = -counts + np.random.random(size=counts.shape)/10
    subs = np.argsort(sort)[:maxBoards]
    allCorners = [allCorners[ix] for ix in subs]
    allIds = [allIds[ix] for ix in subs]
    return allCorners, allIds


def reformat_corners(allCorners, allIds):
    markerCounter = np.array([len(cs) for cs in allCorners])
    allCornersConcat = itertools.chain.from_iterable(allCorners)
    allIdsConcat = itertools.chain.from_iterable(allIds)

    allCornersConcat = np.array(list(allCornersConcat))
    allIdsConcat = np.array(list(allIdsConcat))

    return allCornersConcat, allIdsConcat, markerCounter


def calibrate_aruco(allCornersConcat, allIdsConcat, markerCounter, board, video_params):

    print("calibrating...")
    tstart = time()

    cameraMat = np.eye(3)
    distCoeffs = np.zeros(5)
    dim = (video_params['width'], video_params['height'])
    calib_flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 + \
        cv2.CALIB_FIX_PRINCIPAL_POINT
    error, cameraMat, distCoeffs, rvecs, tvecs = aruco.calibrateCameraAruco(
        allCornersConcat, allIdsConcat, markerCounter, board,
        dim, cameraMat, distCoeffs,
        flags=calib_flags)

    tend = time()
    tdiff = tend - tstart
    print("calibration took {} minutes and {:.1f} seconds".format(
        int(tdiff/60), tdiff-int(tdiff/60)*60))

    out = dict()
    out['error'] = error
    out['camera_mat'] = cameraMat.tolist()
    out['dist_coeff'] = distCoeffs.tolist()
    out['width'] = video_params['width']
    out['height'] = video_params['height']
    out['fps'] = video_params['fps']

    return out

def calibrate_charuco(allCorners, allIds, board, video_params):

    print("calibrating...")
    tstart = time()

    cameraMat = np.eye(3)
    distCoeffs = np.zeros(5)
    dim = (video_params['width'], video_params['height'])
    calib_flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 + \
        cv2.CALIB_FIX_PRINCIPAL_POINT

    error, cameraMat, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        allCorners, allIds, board,
        dim, cameraMat, distCoeffs,
        flags=calib_flags)

    tend = time()
    tdiff = tend - tstart
    print("calibration took {} minutes and {:.1f} seconds".format(
        int(tdiff/60), tdiff-int(tdiff/60)*60))

    out = dict()
    out['error'] = error
    out['camera_mat'] = cameraMat.tolist()
    out['dist_coeff'] = distCoeffs.tolist()
    out['width'] = video_params['width']
    out['height'] = video_params['height']
    out['fps'] = video_params['fps']

    return out


def calibrate_checkerboard(allCorners, board, video_params):

    print("calibrating...")
    tstart = time()

    objpoints = [np.copy(board.objPoints) for _ in allCorners]
    objpoints = np.array(objpoints, dtype='float32')

    allCorners = np.array(allCorners, dtype='float32')

    cameraMat = np.eye(3)
    distCoeffs = np.zeros(5)
    dim = (video_params['width'], video_params['height'])
    calib_flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 + \
        cv2.CALIB_FIX_PRINCIPAL_POINT

    error, cameraMat, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, allCorners, dim, None, None,
        flags=calib_flags)

    tend = time()
    tdiff = tend - tstart
    print("calibration took {} minutes and {:.1f} seconds".format(
        int(tdiff/60), tdiff-int(tdiff/60)*60))

    out = dict()
    out['error'] = error
    out['camera_mat'] = cameraMat.tolist()
    out['dist_coeff'] = distCoeffs.tolist()
    out['width'] = video_params['width']
    out['height'] = video_params['height']
    out['fps'] = video_params['fps']

    return out

def calibrate_camera_aruco(fnames, board):
    allCorners = []
    allIds = []

    board_size = get_board_size(board)
    board_type = get_board_type(board)
    video_params = get_video_params(fnames[0])

    for fname in fnames:
        someCorners, someIds = get_corners_aruco(fname, board)
        allCorners.extend(someCorners)
        allIds.extend(someIds)

    allCorners, allIds = trim_corners(allCorners, allIds, maxBoards=100)
    allCornersConcat, allIdsConcat, markerCounter = reformat_corners(allCorners, allIds)

    print()

    expected_markers = get_expected_corners(board)

    print("found {} markers, {} boards, {} complete boards".format(
        len(allCornersConcat), len(markerCounter),
        np.sum(markerCounter == expected_markers)))

    if board_type == 'charuco':
        calib_params = calibrate_charuco(allCorners, allIds, board, video_params)
    else:
        calib_params = calibrate_aruco(allCornersConcat, allIdsConcat,
                                       markerCounter, board, video_params)

    return calib_params

def calibrate_camera_checkerboard(fnames, board):
    video_params = get_video_params(fnames[0])

    allCorners = []
    allScores = []

    for fname in fnames:
        corners, scores = get_corners_checkerboard(fname, board)
        allCorners.extend(corners)
        allScores.extend(scores)

    allCorners = np.array(allCorners)
    allScores = np.array(allScores)

    n_sub = 200
    if len(allCorners) > n_sub:
        good = np.argsort(allScores)[:n_sub]
        allCorners = allCorners[good]
        allScores = allScores[good]

    print('found {} checkerboard grids'.format(len(allCorners)))

    calib_params = calibrate_checkerboard(allCorners, board, video_params)

    return calib_params


def calibrate_camera(fnames, board):
    board_type = get_board_type(board)
    if board_type == 'checkerboard':
        return calibrate_camera_checkerboard(fnames, board)
    else:
        return calibrate_camera_aruco(fnames, board)


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

    cam_names = [get_cam_name(config, vid) for vid in videos]
    cam_names = sorted(set(cam_names))

    cam_videos = defaultdict(list)

    for vid in videos:
        cname = get_cam_name(config, vid)
        cam_videos[cname].append(vid)

    board = get_calibration_board(config)

    for cname in cam_names:
        fnames = cam_videos[cname]
        outname_base = 'intrinsics_{}.toml'.format(cname)
        outdir = os.path.join(calibration_path, pipeline_calibration_results)
        os.makedirs(outdir, exist_ok=True)
        outname = os.path.join(outdir, outname_base)
        print(outname)
        if os.path.exists(outname):
            continue
        else:
            calib = calibrate_camera(fnames, board)
            with open(outname, 'w') as f:
                toml.dump(calib, f)


calibrate_intrinsics_all = make_process_fun(process_session)
