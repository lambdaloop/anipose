#!/usr/bin/env python3

import cv2
from cv2 import aruco
from tqdm import trange
import numpy as np
import os, os.path
from glob import glob
from collections import defaultdict, Counter
import toml
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten
import queue

from checkerboard import detect_checkerboard

from .common import \
    find_calibration_folder, make_process_fun, \
    get_cam_name, get_video_name, load_intrinsics, \
    get_calibration_board, get_board_type

from .triangulate import triangulate_optim, triangulate_simple, \
    reprojection_error, reprojection_error_und

def fill_points(corners, ids, board):
    board_type = get_board_type(board)

    if board_type == 'checkerboard':
        if corners is not None:
            return corners.reshape(-1, 2)
        else:
            return np.copy(board.objPoints)[:, :2]*np.nan
    else:
        # TODO: this should change with aruco calibration board config
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


def reconstruct_checkerboard(row, camera_mats, camera_mats_dist):
    cam_names = sorted(row.keys())
    mats = [camera_mats[name] for name in cam_names]

    num_points = row[cam_names[0]].shape[0]
    p3ds = []
    errors = []
    for i in range(num_points):
        pts = [row[name][i] for name in cam_names]
        pts = np.array(pts).reshape(-1, 2)
        p3d = triangulate_simple(pts, mats)
        error = reprojection_error_und(p3d, pts, mats, camera_mats_dist)
        p3ds.append(p3d)
        errors.append(error)
    p3ds = np.array(p3ds)
    errors = np.array(errors)
    return p3ds, errors


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

    if intrinsics is None:
        INTRINSICS_K = INTRINSICS_D = None
    else:
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

def estimate_pose_aruco(gray, intrinsics, board):

    detectedCorners, detectedIds = detect_aruco(gray, intrinsics, board)
    if len(detectedIds) < 3:
        return False, None

    INTRINSICS_K = np.array(intrinsics['camera_mat'])
    INTRINSICS_D = np.array(intrinsics['dist_coeff'])

    ret, rvec, tvec = aruco.estimatePoseBoard(detectedCorners, detectedIds, board,
                                              INTRINSICS_K, INTRINSICS_D)

    return True, (detectedCorners, detectedIds, rvec, tvec)

def estimate_pose_checkerboard(grayf, intrinsics, board):
    ratio = 400.0/grayf.shape[0]
    gray = cv2.resize(grayf, (0,0), fx=ratio, fy=ratio,
                      interpolation=cv2.INTER_CUBIC)

    board_size = board.getChessboardSize()

    corners, check_score = detect_checkerboard(gray, board_size)

    if corners is None:
        return False, None

    INTRINSICS_K = np.array(intrinsics['camera_mat'])
    INTRINSICS_D = np.array(intrinsics['dist_coeff'])

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        board.objPoints, corners,
        INTRINSICS_K, INTRINSICS_D,
        confidence=0.9, reprojectionError=10)

    return True, (corners, None, rvec, tvec)


def estimate_pose(gray, intrinsics, board):
    board_type = get_board_type(board)
    if board_type == 'checkerboard':
        return estimate_pose_checkerboard(gray, intrinsics, board)
    else:
        return estimate_pose_aruco(gray, intrinsics, board)


def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(rvec)
    out[:3,:3] = rotmat
    out[:3, 3] = tvec.flatten()
    out[3, 3] = 1
    return out


def get_most_common(vals):
    Z = linkage(whiten(vals), 'ward')
    n_clust = max(len(vals)/10, 3)
    clusts = fcluster(Z, t=n_clust, criterion='maxclust')
    cc = Counter(clusts[clusts >= 0])
    most = cc.most_common(n=1)
    top = most[0][0]
    good = clusts == top
    return good

def select_matrices(Ms):
    Ms = np.array(Ms)
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in Ms]
    tvecs = np.array([M[:3, 3] for M in Ms])
    best = get_most_common(np.hstack([rvecs, tvecs]))
    Ms_best = Ms[best]
    return Ms_best


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
    all_corners = []
    all_ids = []

    for framenum in trange(minlen, ncols=70):
        M_dict = dict()
        corner_dict = dict()
        id_dict = dict()

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
            corner_dict[cam_name] = corners
            id_dict[cam_name] = ids

        if len(M_dict) >= 2:
            go = skip
            all_Ms.append(M_dict)
            all_corners.append(corner_dict)
            all_ids.append(id_dict)

        go = max(0, go-1)

    for cam_name, cap in caps.items():
        cap.release()

    return all_Ms, all_corners, all_ids


def get_transform(matrix_list, left, right):
    L = []
    for d in matrix_list:
        if left in d and right in d:
            M = np.matmul(d[left], np.linalg.inv(d[right]))
            L.append(M)
    L_best = select_matrices(L)
    M_mean = mean_transform(L_best)
    # M_mean = mean_transform_robust(L, M_mean, error=0.5)
    # M_mean = mean_transform_robust(L, M_mean, error=0.2)
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


def get_calibration_graph(imgpoints, cam_names):
    n_cams = len(cam_names)
    connections = defaultdict(int)

    for p in imgpoints:
        keys = sorted(p.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a = keys[i]
                b = keys[j]
                connections[(a,b)] += 1
                connections[(b,a)] += 1

    components = dict(zip(cam_names, range(n_cams)))
    edges = set(connections.items())

    print(edges)

    graph = defaultdict(list)

    for edgenum in range(n_cams-1):
        if len(edges) == 0:
            return None

        (a, b), weight = max(edges, key=lambda x: x[1])
        graph[a].append(b)
        graph[b].append(a)

        match = components[a]
        replace = components[b]
        for k, v in components.items():
            if match == v:
                components[k] = replace

        for e in edges.copy():
            (a,b), w = e
            if components[a] == components[b]:
                edges.remove(e)

    return graph

def find_calibration_pairs(graph, source=0):
    pairs = []
    explored = set()

    q = queue.deque()
    q.append(source)

    while len(q) > 0:
        item = q.pop()
        explored.add(item)

        for new in graph[item]:
            if new not in explored:
                q.append(new)
                pairs.append( (item, new) )
    return pairs

def compute_camera_matrices(matrix_list, pairs, source=0):
    extrinsics = dict()
    extrinsics[source] = np.identity(4)
    for (a,b) in pairs:
        ext = get_transform(matrix_list, b, a)
        extrinsics[b] = np.matmul(ext, extrinsics[a])
    return extrinsics

def get_extrinsics(fname_dicts, cam_intrinsics, cam_align, board, skip=20):
    ## TODO optimize transforms based on reprojection errors
    ## TODO build up camera matrices based on pairs
    matrix_list = []
    corner_list = []
    id_list = []
    cam_names = set()
    for fd in fname_dicts:
        ml, corners, ids = get_matrices(fd, cam_intrinsics, board, skip=skip)
        matrix_list.extend(ml)
        corner_list.extend(corners)
        id_list.extend(ids)
        cam_names.update(fd.keys())

    cam_names = sorted(cam_names)
    
    # pairs = get_all_matrix_pairs(matrix_list, sorted(cam_names))
    graph = get_calibration_graph(matrix_list, cam_names)
    pairs = find_calibration_pairs(graph, source=cam_align)
    extrinsics = compute_camera_matrices(matrix_list, pairs, source=cam_align)

    return extrinsics

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
    cam_align = config['triangulation']['cam_align']

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

        extrinsics = get_extrinsics(fname_dicts, intrinsics, cam_align, board)
        extrinsics_out = {}
        for k, v in extrinsics.items():
            extrinsics_out[k] = v.tolist()

        with open(outname, 'w') as f:
            toml.dump(extrinsics_out, f)


calibrate_extrinsics_all = make_process_fun(process_session)
