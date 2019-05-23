#!/usr/bin/env python3

import cv2
from cv2 import aruco
from tqdm import trange
# import numpy as np
import os, os.path
from glob import glob
from collections import defaultdict, Counter
import toml
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten
import queue
from scipy import optimize
from time import time
import autograd.numpy as np
from autograd import grad
from autograd.misc import optimizers

from checkerboard import detect_checkerboard

from .common import \
    find_calibration_folder, make_process_fun, \
    get_cam_name, get_video_name, load_intrinsics, \
    get_calibration_board, get_board_type, get_expected_corners

from .triangulate import triangulate_optim, triangulate_simple, \
    reprojection_error, reprojection_error_und

from autograd.extend import primitive, defvjp

@primitive
def apad(array, width, mode, **kwargs):
    return np.pad(array, width, mode, **kwargs)

def _unpad(array, width):
    if np.isscalar(width):
        width = [[width, width]]
    elif np.shape(width) == (1,):
        width = [np.concatenate((width, width))]
    elif np.shape(width) == (2,):
        width = [width]
    if np.shape(width)[0] == 1:
        width = np.repeat(width, np.ndim(array), 0)
    idxs = tuple(slice(l, -u or None) for l, u in width)
    return array[idxs]

def pad_vjp(ans, array, pad_width, mode, **kwargs):
    assert mode == "constant", "Only constant mode padding is supported."
    return lambda g: _unpad(g, pad_width)

defvjp(apad, pad_vjp)


def fill_points(corners, ids, board):
    board_type = get_board_type(board)

    if board_type == 'checkerboard':
        if corners is not None:
            return corners.reshape(-1, 2)
        else:
            return np.copy(board.objPoints)[:, :2]*np.nan
    elif board_type == 'aruco':
        num_corners = get_expected_corners(board)

        # N boxes with 4 corners each
        out = np.zeros((num_corners*4, 2))
        out.fill(np.nan)

        if corners is None or ids is None:
            return out

        for id_wrap, corner_wrap in zip(ids, corners):
            ix = id_wrap[0]
            corner = corner_wrap.flatten().reshape(4, 2)
            if ix >= num_corners:
                continue
            out[ix*4:(ix+1)*4,:] = corner

        return out
    elif board_type == 'charuco':
        num_corners = get_expected_corners(board)

        out = np.zeros((num_corners, 2))
        out.fill(np.nan)

        if ids is None or corners is None:
            return out

        corners = corners.reshape(-1, 2)
        ids = ids.flatten()

        for ix, corner in zip(ids, corners):
            out[ix] = corner

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
    board_type = get_board_type(board)

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
        aruco.refineDetectedMarkers(gray, board, corners, ids,
                                    rejectedImgPoints,
                                    INTRINSICS_K, INTRINSICS_D,
                                    parameters=params)

    if board_type == 'charuco' and len(detectedCorners) > 0:
        ret, detectedCorners, detectedIds = aruco.interpolateCornersCharuco(
            detectedCorners, detectedIds, gray, board)

        if detectedIds is None:
            detectedCorners = detectedIds = []

    return detectedCorners, detectedIds

def estimate_pose_aruco(gray, intrinsics, board):

    detectedCorners, detectedIds = detect_aruco(gray, intrinsics, board)
    if len(detectedIds) < 3:
        return False, None

    INTRINSICS_K = np.array(intrinsics['camera_mat'])
    INTRINSICS_D = np.array(intrinsics['dist_coeff'])

    board_type = get_board_type(board)

    if board_type == 'charuco':
        ret, rvec, tvec = aruco.estimatePoseCharucoBoard(
            detectedCorners, detectedIds, board, INTRINSICS_K, INTRINSICS_D)
    else:
        ret, rvec, tvec = aruco.estimatePoseBoard(
            detectedCorners, detectedIds, board, INTRINSICS_K, INTRINSICS_D)

    if not ret or rvec is None or tvec is None:
        return False, None

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

def get_rtvec(M):
    rvec = cv2.Rodrigues(M[:3, :3])[0].flatten()
    tvec = M[:3, 3].flatten()
    return rvec, tvec

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


def get_matrices(fname_dict, cam_intrinsics, board, skip=40):
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
    all_points = []

    for framenum in trange(minlen, ncols=70):
        M_dict = dict()
        point_dict = dict()

        for cam_name in cam_names:
            cap = caps[cam_name]
            ret, frame = cap.read()

            if framenum % skip != 0 and go <= 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            intrinsics = cam_intrinsics[cam_name]
            success, result = estimate_pose(gray, intrinsics, board)
            if not success:
                continue

            corners, ids, rvec, tvec = result
            M_dict[cam_name] = make_M(rvec, tvec)

            points = fill_points(corners, ids, board)
            points_flat = points.reshape(-1, 1, 2)
            points_new = cv2.undistortPoints(
                points_flat,
                np.array(intrinsics['camera_mat']),
                np.array(intrinsics['dist_coeff']))

            point_dict[cam_name] = points_new.reshape(points.shape)


        if len(M_dict) >= 2:
            go = skip
            all_Ms.append(M_dict)
            all_points.append(point_dict)

        go = max(0, go-1)

    for cam_name, cap in caps.items():
        cap.release()

    return all_Ms, all_points


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

    print(sorted(edges))

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

def estimate_calibration_errors(point_list, intrinsics, extrinsics):
    errors = []
    for points in point_list:
        cnames = points.keys()
        cam_mats = np.array([extrinsics[c] for c in cnames])
        cam_mats_dist = np.array([intrinsics[c]['camera_mat'] for c in cnames])
        pts = np.array([points[c] for c in cnames])
        for i in range(pts.shape[1]):
            if np.sum(~np.isnan(pts[:, i, 0])) < 2:
                continue

            good = ~np.isnan(pts[:, i, 0])
            p3d = triangulate_simple(pts[good, i], cam_mats[good])
            error = reprojection_error_und(p3d, pts[good, i], cam_mats[good], cam_mats_dist[good])
            errors.append(error)

    return np.array(errors)


def rodrigues_vec(angles):
    theta = np.linalg.norm(angles)
    r = angles / theta

    mat = [[0, -r[2], r[1]],
           [r[2], 0, -r[0]],
           [-r[1], r[0], 0]]
    mat = np.array(mat)

    R = np.cos(theta) * np.eye(3) + \
        (1-np.cos(theta))*np.outer(r, r) + \
        np.sin(theta) * mat

    return R


def make_M_vec(rvec, tvec):
    #rotmat, _ = cv2.Rodrigues(rvec)
    rotmat = rodrigues_vec(rvec.flatten())
    rotmatm = apad(rotmat, [(0, 1), (0, 1)], mode='constant')
    tvecf = np.reshape(tvec.flatten(), (3, 1))
    tvecm = apad(tvecf, [(0, 1), (3, 0)], 'constant')
    const = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    out = rotmatm + tvecm + const
    return out


def mats_to_params(mats):
    params = np.zeros(len(mats)*6)
    for i, M in enumerate(mats):
        rvec, tvec = get_rtvec(M)
        s = i*6
        params[s:s+3] = rvec
        params[s+3:s+6] = tvec
    return params

def params_to_mats(params):
    cam_mats = [np.identity(4)]
    n_cams = len(params) // 6
    for i in range(n_cams):
        s = i*6
        MX = make_M_vec(params[s:s+3], params[s+3:s+6])
        cam_mats.append(MX)
    cam_mats = np.array(cam_mats)
    return cam_mats

def setup_bundle_problem(point_list, intrinsics, extrinsics, cam_align):
    out = []
    # make sure cam_align is first
    cnames = [cam_align] + sorted(set(extrinsics.keys()) - {cam_align})
    cam_mats = np.array([extrinsics[c] for c in cnames])

    points = point_list[0]
    v = list(points.values())[0]
    template = v*np.nan

    for points in point_list:
        keys = sorted(points.keys())
        pts = []
        for c in cnames:
            if c in points:
                p = points[c]
            else:
                p = template
            pts.append(p)
        pts = np.array(pts)
        for i in range(pts.shape[1]):
            if np.sum(~np.isnan(pts[:, i, 0])) < 2:
                continue
            out.append(pts[:, i])

    out = np.array(out)
    return out, cnames, cam_mats


def triangulate_simple_vec(points, camera_mats):
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        row1 = apad(np.reshape(x*mat[2] - mat[0], (1, 4)),
                      [(i*2, num_cams*2 - i*2 - 1), (0, 0)],
                      'constant')
        row2 = apad(np.reshape(y*mat[2] - mat[1], (1, 4)),
                      [(i*2+1, num_cams*2 - i*2 - 2), (0, 0)],
                      'constant')
        A = A + row1
        A = A + row2
        # A[(i*2):(i*2+1)] = x*mat[2]-mat[0]
        # A[(i*2+1):(i*2+2)] = y*mat[2]-mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    p3d = vh[-1]
    p3d = p3d / p3d[3]
    return p3d

def reprojection_error_vec(p3d, points2d, camera_mats):
    proj = np.dot(camera_mats, p3d)
    proj = proj[:, :2] / proj[:, 2, None]
    errors = np.linalg.norm(proj - points2d, axis=1)
    return np.mean(errors)

def evaluate_errors(the_points, params, all_good=None):
    cam_mats = params_to_mats(params)

    if all_good is None:
        all_good = [None]*len(the_points)
        for ptnum, points in enumerate(the_points):
            all_good[ptnum] = ~np.isnan(points[:, 0])

    errors = []
    for ptnum, points in enumerate(the_points):
        good = all_good[ptnum]
        p3d = triangulate_simple_vec(points[good], cam_mats[good])
        err = reprojection_error_vec(p3d, points[good], cam_mats[good])
        errors.append(err)
    return np.array(errors)

def make_error_fun(the_points, n_samples=None, sum=False):
    the_points_sampled = the_points
    if n_samples is not None and n_samples < the_points.shape[0]:
        samples = np.random.choice(the_points.shape[0], n_samples,
                                   replace=False)
        the_points_sampled = the_points[samples]

    all_good = [None]*len(the_points_sampled)
    for ptnum, points in enumerate(the_points_sampled):
        all_good[ptnum] = ~np.isnan(points[:, 0])

    def error_fun(params):
        errors = evaluate_errors(the_points_sampled, params, all_good)
        if sum:
            return np.sum(errors)
        else:
            return errors
    return error_fun, the_points_sampled

def make_grad_fun(the_points, n_samples=3):
    def grad_fun(params, t):
        fun, _ = make_error_fun(the_points, n_samples=n_samples, sum=True)
        gf = grad(fun)
        scale = 1.0 / (1 + 2*t)
        return gf(params) * scale
    return grad_fun

def bundle_adjust(all_points, cam_names, cam_mats):
    params = mats_to_params(cam_mats[1:])

    # error_fun, points_sampled = make_error_fun(all_points, n_samples=500)
    # opt = optimize.least_squares(error_fun, params, loss='linear',
    #                              method='trf', tr_solver='lsmr')
    # best_params = opt.x

    error_fun_print, _ = make_error_fun(all_points, n_samples=50, sum=True)
    def print_step(x, i, g):
        if i % 25 == 0:
            print(i, error_fun_print(x))
    grad_fun = make_grad_fun(all_points, n_samples=1)
    best_params = optimizers.adam(grad_fun, params,
                                  callback=print_step,
                                  step_size=0.001,
                                  num_iters=1000)

    mats_new = params_to_mats(best_params)

    extrinsics_new = dict(zip(cam_names, mats_new))

    return extrinsics_new


def get_extrinsics(fname_dicts, cam_intrinsics, cam_align, board, skip=20):
    ## TODO optimize transforms based on reprojection errors
    ## TODO build up camera matrices based on pairs
    matrix_list = []
    point_list = []
    cam_names = set()
    for fd in fname_dicts:
        ml, points = get_matrices(fd, cam_intrinsics, board, skip=skip)
        matrix_list.extend(ml)
        point_list.extend(points)
        cam_names.update(fd.keys())

    cam_names = sorted(cam_names)

    # pairs = get_all_matrix_pairs(matrix_list, sorted(cam_names))
    graph = get_calibration_graph(matrix_list, cam_names)
    pairs = find_calibration_pairs(graph, source=cam_align)
    extrinsics = compute_camera_matrices(matrix_list, pairs, source=cam_align)

    errors = estimate_calibration_errors(point_list, cam_intrinsics, extrinsics)

    print('before', np.mean(errors))

    all_points, cam_names_new, cam_mats = setup_bundle_problem(
        point_list, cam_intrinsics, extrinsics, cam_align)

    t1 = time()
    extrinsics_new = bundle_adjust(all_points, cam_names_new, cam_mats)
    t2 = time()

    print('bundle adjustment took {:.1f} seconds'.format(t2 - t1))

    errors = estimate_calibration_errors(point_list, cam_intrinsics, extrinsics_new)
    print('after', np.mean(errors))

    return extrinsics_new, np.mean(errors)

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

        extrinsics, error = get_extrinsics(fname_dicts, intrinsics, cam_align, board)
        extrinsics_out = {}
        for k, v in extrinsics.items():
            extrinsics_out[k] = v.tolist()
        extrinsics_out['error'] = float(error)

        with open(outname, 'w') as f:
            toml.dump(extrinsics_out, f)


calibrate_extrinsics_all = make_process_fun(process_session)
