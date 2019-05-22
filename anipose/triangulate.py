#!/usr/bin/env python3

from tqdm import trange
import numpy as np
from collections import defaultdict
import os
import os.path
import pandas as pd
import toml
from numpy import array as arr
from glob import glob
from scipy import optimize
import cv2

from .common import make_process_fun, find_calibration_folder, \
    get_video_name, get_cam_name, natural_keys, \
    load_intrinsics, load_extrinsics


def expand_matrix(mtx):
    z = np.zeros((4,4))
    z[0:3,0:3] = mtx[0:3,0:3]
    z[3,3] = 1
    return z

def reproject_points(p3d, points2d, camera_mats):
    proj = np.dot(camera_mats, p3d)
    proj = proj[:, :2] / proj[:, 2, None]
    return proj


def reprojection_error(p3d, points2d, camera_mats):
    proj = np.dot(camera_mats, p3d)
    proj = proj[:, :2] / proj[:, 2, None]
    errors = np.linalg.norm(proj - points2d, axis=1)
    return np.mean(errors)


def distort_points_cams(points, camera_mats):
    out = []
    for i in range(len(points)):
        point = np.append(points[i], 1)
        mat = camera_mats[i]
        new = mat.dot(point)[:2]
        out.append(new)
    return np.array(out)

def reprojection_error_und(p3d, points2d, camera_mats, camera_mats_dist):
    proj = np.dot(camera_mats, p3d)
    proj = proj[:, :2] / proj[:, 2, None]
    proj_d = distort_points_cams(proj, camera_mats_dist)
    points2d_d = distort_points_cams(points2d, camera_mats_dist)
    errors = np.linalg.norm(proj_d - points2d_d, axis=1)
    return np.mean(errors)

def triangulate_simple(points, camera_mats):
    num_cams = len(camera_mats)
    A = np.zeros((num_cams*2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i*2):(i*2+1)] = x*mat[2]-mat[0]
        A[(i*2+1):(i*2+2)] = y*mat[2]-mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d / p3d[3]
    return p3d

def optim_error_fun(points, camera_mats):
    def fun(x):
        p3d = np.array([x[0], x[1], x[2], 1])
        proj = np.dot(camera_mats, p3d)
        resid = points - proj[:, :2] / proj[:, 2, None]
        # return resid.flatten()
        return np.linalg.norm(resid, axis=1)
    return fun

def triangulate_optim(points, camera_mats, max_error=20):
    try:
        p3d = triangulate_simple(points, camera_mats)
        error = reprojection_error(p3d, points, camera_mats)
    except np.linalg.linalg.LinAlgError:
        return np.array([0,0,0,0])

    fun = optim_error_fun(points, camera_mats)
    try:
        res = optimize.least_squares(fun, p3d[:3])
        x = res.x
        p3d = np.array([x[0], x[1], x[2], 1])
    except ValueError:
        pass

    return p3d


def proj(u, v):
    """Project u onto v"""
    return u * np.dot(v,u) / np.dot(u,u)

def ortho(u, v):
    """Orthagonalize u with respect to v"""
    return u - proj(v, u)

def get_median(all_points_3d, ix):
    pts = all_points_3d[:, ix]
    pts = pts[~np.isnan(pts[:, 0])]
    return np.median(pts, axis=0)


def correct_coordinate_frame(config, all_points_3d, bodyparts):
    """Given a config and a set of points and bodypart names, this function will rotate the coordinate frame to match the one in config"""
    bp_index = dict(zip(bodyparts, range(len(bodyparts))))
    axes_mapping = dict(zip('xyz', range(3)))

    ref_point = config['triangulation']['reference_point']
    axes_spec = config['triangulation']['axes']
    a_dirx, a_l, a_r = axes_spec[0]
    b_dirx, b_l, b_r = axes_spec[1]

    a_dir = axes_mapping[a_dirx]
    b_dir = axes_mapping[b_dirx]

    ## find the missing direction
    done = np.zeros(3, dtype='bool')
    done[a_dir] = True
    done[b_dir] = True
    c_dir = np.where(~done)[0][0]

    a_lv = get_median(all_points_3d, bp_index[a_l])
    a_rv = get_median(all_points_3d, bp_index[a_r])
    b_lv = get_median(all_points_3d, bp_index[b_l])
    b_rv = get_median(all_points_3d, bp_index[b_r])

    a_diff = a_rv - a_lv
    b_diff = ortho(b_rv - b_lv, a_diff)

    M = np.zeros((3,3))
    M[a_dir] = a_diff
    M[b_dir] = b_diff
    M[c_dir] = np.cross(a_diff, b_diff)

    M /= np.linalg.norm(M, axis=1)[:,None]

    center = get_median(all_points_3d, bp_index[ref_point])

    all_points_3d_adj = (all_points_3d - center).dot(M.T)
    center_new = get_median(all_points_3d_adj, bp_index[ref_point])
    all_points_3d_adj = all_points_3d_adj - center_new

    return all_points_3d_adj

def triangulate(config,
                calib_folder, video_folder, pose_folder,
                fname_dict, output_fname):

    ## TODO: make the recorder.toml file configurable
    record_fname = os.path.join(video_folder, 'recorder.toml')

    if os.path.exists(record_fname):
        record_dict = toml.load(record_fname)
    else:
        record_dict = None
        # if 'cameras' not in config:
        # ## TODO: more detailed error?
        #     print("-- no crop windows found")
        #     return

    cam_names, pose_names = list(zip(*sorted(fname_dict.items())))

    intrinsics = load_intrinsics(calib_folder, cam_names)
    extrinsics = load_extrinsics(calib_folder)

    offsets_dict = dict()
    for cname in cam_names:
        if record_dict is None:
            if 'cameras' not in config or cname not in config['cameras']:
                # print("W: no crop window found for camera {}, assuming no crop".format(cname))
                offsets_dict[cname] = [0, 0]
            else:
                offsets_dict[cname] = config['cameras'][cname]['offset']
        else:
            offsets_dict[cname] = record_dict['cameras'][cname]['video']['ROIPosition']

    offsets = []
    cam_mats = []
    cam_mats_dist = []

    for cname in cam_names:
        mat = arr(extrinsics[cname])
        left = arr(intrinsics[cname]['camera_mat'])
        cam_mats.append(mat)
        cam_mats_dist.append(left)
        offsets.append(offsets_dict[cname])

    offsets = arr(offsets)
    cam_mats = arr(cam_mats)
    cam_mats_dist = arr(cam_mats_dist)

    maxlen = 0
    for pose_name in pose_names:
        dd = pd.read_hdf(pose_name)
        length = len(dd.index)
        maxlen = max(maxlen, length)

    length = maxlen
    dd = pd.read_hdf(pose_names[0])
    scorer = dd.columns.levels[0][0]
    dd = dd[scorer]

    bodyparts = arr(dd.columns.levels[0])

    # frame, camera, bodypart, xy
    all_points_raw = np.zeros((length, len(cam_names), len(bodyparts), 2))
    all_points_und = np.zeros((length, len(cam_names), len(bodyparts), 2))
    all_scores = np.zeros((length, len(cam_names), len(bodyparts)))

    for ix_cam, (cam_name, pose_name, offset) in \
        enumerate(zip(cam_names, pose_names, offsets)):
        dd = pd.read_hdf(pose_name)
        scorer = dd.columns.levels[0][0]
        dd = dd[scorer]

        index = arr(dd.index)
        for ix_bp, bp in enumerate(bodyparts):
            X = arr(dd[bp])
            all_points_raw[index, ix_cam, ix_bp, :] = X[:, :2] + [offset[0], offset[1]]
            all_scores[index, ix_cam, ix_bp] = X[:, 2]

        calib = intrinsics[cam_name]
        points = all_points_raw[:, ix_cam].reshape(-1, 1, 2)
        points_new = cv2.undistortPoints(
            points, arr(calib['camera_mat']), arr(calib['dist_coeff']))
        all_points_und[:, ix_cam] = points_new.reshape(all_points_raw[:, ix_cam].shape)


    shape = all_points_raw.shape

    all_points_3d = np.zeros((shape[0], shape[2], 3))
    all_points_3d.fill(np.nan)

    errors = np.zeros((shape[0], shape[2]))
    errors.fill(np.nan)

    scores_3d = np.zeros((shape[0], shape[2]))
    scores_3d.fill(np.nan)

    num_cams = np.zeros((shape[0], shape[2]))
    num_cams.fill(np.nan)

    # TODO: configure this threshold
    all_points_und[all_scores < 0.3] = np.nan

    for i in trange(all_points_und.shape[0], ncols=70):
        for j in range(all_points_und.shape[2]):
            pts = all_points_und[i, :, j, :]
            good = ~np.isnan(pts[:, 0])
            if np.sum(good) >= 2:
                # TODO: make triangulation type configurable
                # p3d = triangulate_optim(pts[good], cam_mats[good])
                p3d = triangulate_simple(pts[good], cam_mats[good])
                all_points_3d[i, j] = p3d[:3]
                errors[i,j] = reprojection_error_und(p3d, pts[good], cam_mats[good], cam_mats_dist[good])
                num_cams[i,j] = np.sum(good)
                scores_3d[i,j] = np.min(all_scores[i, :, j][good])

    all_points_3d_adj = correct_coordinate_frame(config, all_points_3d, bodyparts)

    dout = pd.DataFrame()
    for bp_num, bp in enumerate(bodyparts):
        for ax_num, axis in enumerate(['x','y','z']):
            dout[bp + '_' + axis] = all_points_3d_adj[:, bp_num, ax_num]
        dout[bp + '_error'] = errors[:, bp_num]
        dout[bp + '_ncams'] = num_cams[:, bp_num]
        dout[bp + '_score'] = scores_3d[:, bp_num]

    dout['fnum'] = np.arange(length)

    dout.to_csv(output_fname, index=False)


def process_session(config, session_path):
    pipeline_videos_raw = config['pipeline']['videos_raw']
    pipeline_calibration_results = config['pipeline']['calibration_results']
    pipeline_pose = config['pipeline']['pose_2d']
    pipeline_pose_filter = config['pipeline']['pose_2d_filter']
    pipeline_3d = config['pipeline']['pose_3d']

    calibration_path = find_calibration_folder(config, session_path)
    if calibration_path is None:
        return

    if config['filter']['enabled']:
        pose_folder = os.path.join(session_path, pipeline_pose_filter)
    else:
        pose_folder = os.path.join(session_path, pipeline_pose)

    calib_folder = os.path.join(calibration_path, pipeline_calibration_results)
    video_folder = os.path.join(session_path, pipeline_videos_raw)
    output_folder = os.path.join(session_path, pipeline_3d)

    pose_files = glob(os.path.join(pose_folder, '*.h5'))

    cam_videos = defaultdict(list)

    for pf in pose_files:
        name = get_video_name(config, pf)
        cam_videos[name].append(pf)

    vid_names = cam_videos.keys()
    vid_names = sorted(vid_names, key=natural_keys)

    if len(vid_names) > 0:
        os.makedirs(output_folder, exist_ok=True)
    
    fname_dicts = []
    for name in vid_names:
        print(name)
        fnames = cam_videos[name]
        cam_names = [get_cam_name(config, f) for f in fnames]
        fname_dict = dict(zip(cam_names, fnames))
        fname_dicts.append(fname_dict)

        output_fname = os.path.join(output_folder, name + '.csv')

        if os.path.exists(output_fname):
            continue

        triangulate(config,
                    calib_folder, video_folder, pose_folder,
                    fname_dict, output_fname)


triangulate_all = make_process_fun(process_session)
