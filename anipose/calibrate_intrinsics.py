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

from .common import make_process_fun, get_cam_name, find_calibration_folder, \
    get_video_params, get_calibration_board

def get_corners(fname, board):
    cap = cv2.VideoCapture(fname)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    allCorners = []
    allIds = []

    for i in trange(length, ncols=70):
        ret, frame = cap.read()
        if not ret:
            break

        if i % 10 != 0:
            continue

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        grayb = gray

        params = aruco.DetectorParameters_create()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshWinSizeMin = 100
        params.adaptiveThreshWinSizeMax = 700
        params.adaptiveThreshWinSizeStep = 50
        params.adaptiveThreshConstant = 5

        corners, ids, rejectedImgPoints = aruco.detectMarkers(grayb, board.dictionary, parameters=params)

        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = aruco.refineDetectedMarkers(grayb, board, corners, ids, rejectedImgPoints, parameters=params)

        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if len(detectedCorners) >= 2:
            allCorners.append(detectedCorners)
            allIds.append(detectedIds)


    cap.release()

    return allCorners, allIds


def trim_corners(allCorners, allIds, maxBoards=85):
    counts = np.array([len(cs) for cs in allCorners])
    sort = -counts + np.random.random(size=counts.shape)/2
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
    error, cameraMat, distCoeffs, rvecs, tvecs = aruco.calibrateCameraAruco(allCornersConcat, allIdsConcat, markerCounter, board, dim, cameraMat, distCoeffs)

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

def calibrate_camera(fnames, board):
    allCorners = []
    allIds = []

    board_size = board.getGridSize()

    video_params = get_video_params(fnames[0])

    for fname in fnames:
        someCorners, someIds = get_corners(fname, board)
        allCorners.extend(someCorners)
        allIds.extend(someIds)

    allCorners, allIds = trim_corners(allCorners, allIds, maxBoards=85)
    allCornersConcat, allIdsConcat, markerCounter = reformat_corners(allCorners, allIds)

    print()

    print("found {} markers, {} boards, {} complete boards".format(
        len(allCornersConcat), len(markerCounter),
        np.sum(markerCounter == board_size[0]*board_size[1])))

    calib_params = calibrate_aruco(allCornersConcat, allIdsConcat,
                                   markerCounter, board, video_params)

    return calib_params


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
