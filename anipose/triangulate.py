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
    get_video_name, get_cam_name, natural_keys

from aniposelib.cameras import CameraGroup

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
    # form a right handed coordinate system
    if (a_dir,b_dir) in [(0,1), (2,0), (1,2)]:
        M[c_dir] = np.cross(a_diff, b_diff)
    else:
        M[c_dir] = np.cross(b_diff, a_diff)

    M /= np.linalg.norm(M, axis=1)[:,None]

    center = get_median(all_points_3d, bp_index[ref_point])

    all_points_3d_adj = all_points_3d.dot(M.T)
    center_new = get_median(all_points_3d_adj, bp_index[ref_point])
    all_points_3d_adj = all_points_3d_adj - center_new

    return all_points_3d_adj, M, center_new


def load_pose2d_fnames(fname_dict, offsets_dict=None, cam_names=None):
    if cam_names is None:
        cam_names = sorted(fname_dict.keys())
    pose_names = [fname_dict[cname] for cname in cam_names]

    if offsets_dict is None:
        offsets_dict = dict([(cname, (0,0)) for cname in cam_names])

    datas = []
    for ix_cam, (cam_name, pose_name) in \
            enumerate(zip(cam_names, pose_names)):
        dlabs = pd.read_hdf(pose_name)
        if len(dlabs.columns.levels) > 2:
            scorer = dlabs.columns.levels[0][0]
            dlabs = dlabs.loc[:, scorer]

        bp_index = dlabs.columns.names.index('bodyparts')
        joint_names = list(dlabs.columns.get_level_values(bp_index).unique())
        dx = offsets_dict[cam_name][0]
        dy = offsets_dict[cam_name][1]

        for joint in joint_names:
            dlabs.loc[:, (joint, 'x')] += dx
            dlabs.loc[:, (joint, 'y')] += dy

        datas.append(dlabs)

    n_cams = len(cam_names)
    n_joints = len(joint_names)
    n_frames = min([d.shape[0] for d in datas])

    # frame, camera, bodypart, xy
    points = np.full((n_cams, n_frames, n_joints, 2), np.nan, 'float')
    scores = np.full((n_cams, n_frames, n_joints), np.zeros(1), 'float')#initialise as zeros, instead of NaN, makes more sense? 

    for cam_ix, dlabs in enumerate(datas):
        for joint_ix, joint_name in enumerate(joint_names):
            try:
                points[cam_ix, :, joint_ix] = np.array(dlabs.loc[:, (joint_name, ('x', 'y'))])[:n_frames] 
                scores[cam_ix, :, joint_ix] = np.array(dlabs.loc[:, (joint_name, ('likelihood'))])[:n_frames].ravel()
            except KeyError:
                pass

    return {
        'cam_names': cam_names,
        'points': points,
        'scores': scores,
        'bodyparts': joint_names
    }


def load_offsets_dict(config, cam_names, video_folder=None):
    ## TODO: make the recorder.toml file configurable
    # record_fname = os.path.join(video_folder, 'recorder.toml')

    # if os.path.exists(record_fname):
    #     record_dict = toml.load(record_fname)
    # else:
    #     record_dict = None
    #     # if 'cameras' not in config:
    #     # ## TODO: more detailed error?
    #     #     print("-- no crop windows found")
    #     #     return

    offsets_dict = dict()
    for cname in cam_names:
        # if record_dict is None:
        if 'cameras' not in config or cname not in config['cameras']:
            # print("W: no crop window found for camera {}, assuming no crop".format(cname))
            offsets_dict[cname] = (0, 0)
        else:
            offsets_dict[cname] = tuple(config['cameras'][cname]['offset'])
        # else:
        #     offsets_dict[cname] = record_dict['cameras'][cname]['video']['ROIPosition']

    return offsets_dict

def load_constraints(config, bodyparts, key='constraints'):
    constraints_names = config['triangulation'].get(key, [])
    bp_index = dict(zip(bodyparts, range(len(bodyparts))))
    constraints = []
    for a, b in constraints_names:
        assert a in bp_index, 'Bodypart {} from constraints not found in list of bodyparts'.format(a)
        assert b in bp_index, 'Bodypart {} from constraints not found in list of bodyparts'.format(b)
        con = [bp_index[a], bp_index[b]]
        constraints.append(con)
    return constraints


def triangulate(config,
                calib_folder, video_folder, pose_folder,
                fname_dict, output_fname):

    cam_names = sorted(fname_dict.keys())

    calib_fname = os.path.join(calib_folder, 'calibration.toml')
    cgroup = CameraGroup.load(calib_fname)

    offsets_dict = load_offsets_dict(config, cam_names, video_folder)

    out = load_pose2d_fnames(fname_dict, offsets_dict, cam_names)
    all_points_raw = out['points']
    all_scores = out['scores']
    bodyparts = out['bodyparts']

    cgroup = cgroup.subset_cameras_names(cam_names)

    n_cams, n_frames, n_joints, _ = all_points_raw.shape

    bad = all_scores < config['triangulation']['score_threshold']
    all_points_raw[bad] = np.nan

    if config['triangulation']['optim']:
        constraints = load_constraints(config, bodyparts)
        constraints_weak = load_constraints(config, bodyparts, 'constraints_weak')

        points_2d = all_points_raw
        scores_2d = all_scores

        points_shaped = points_2d.reshape(n_cams, n_frames*n_joints, 2)
        if config['triangulation']['ransac']:
            points_3d_init, _, _, _ = cgroup.triangulate_ransac(points_shaped, progress=True)
        else:
            points_3d_init = cgroup.triangulate(points_shaped, progress=True)
        points_3d_init = points_3d_init.reshape((n_frames, n_joints, 3))

        c = np.isfinite(points_3d_init[:, :, 0])
        if np.sum(c) < 20:
            print("warning: not enough 3D points to run optimization")
            points_3d = points_3d_init
        else:
            points_3d = cgroup.optim_points(
                points_2d, points_3d_init,
                constraints=constraints,
                constraints_weak=constraints_weak,
                # scores=scores_2d,
                scale_smooth=config['triangulation']['scale_smooth'],
                scale_length=config['triangulation']['scale_length'],
                scale_length_weak=config['triangulation']['scale_length_weak'],
                n_deriv_smooth=config['triangulation']['n_deriv_smooth'],
                reproj_error_threshold=config['triangulation']['reproj_error_threshold'],
                verbose=True)

        points_2d_flat = points_2d.reshape(n_cams, -1, 2)
        points_3d_flat = points_3d.reshape(-1, 3)

        errors = cgroup.reprojection_error(
            points_3d_flat, points_2d_flat, mean=True)
        good_points = ~np.isnan(all_points_raw[:, :, :, 0])
        num_cams = np.sum(good_points, axis=0).astype('float')

        all_points_3d = points_3d
        all_errors = errors.reshape(n_frames, n_joints)

        all_scores[~good_points] = 2
        scores_3d = np.min(all_scores, axis=0)

        scores_3d[num_cams < 1] = np.nan
        all_errors[num_cams < 1] = np.nan

    else:
        points_2d = all_points_raw.reshape(n_cams, n_frames*n_joints, 2)
        if config['triangulation']['ransac']:
            points_3d, picked, p2ds, errors = cgroup.triangulate_ransac(
                points_2d, min_cams=3, progress=True)

            all_points_picked = p2ds.reshape(n_cams, n_frames, n_joints, 2)
            good_points = ~np.isnan(all_points_picked[:, :, :, 0])

            num_cams = np.sum(np.sum(picked, axis=0), axis=1)\
                         .reshape(n_frames, n_joints)\
                         .astype('float')
        else:
            points_3d = cgroup.triangulate(points_2d, progress=True)
            errors = cgroup.reprojection_error(points_3d, points_2d, mean=True)
            good_points = ~np.isnan(all_points_raw[:, :, :, 0])
            num_cams = np.sum(good_points, axis=0).astype('float')

        all_points_3d = points_3d.reshape(n_frames, n_joints, 3)
        all_errors = errors.reshape(n_frames, n_joints)

        all_scores[~good_points] = 2
        scores_3d = np.min(all_scores, axis=0)

        scores_3d[num_cams < 2] = np.nan
        all_errors[num_cams < 2] = np.nan
        num_cams[num_cams < 2] = np.nan

    if 'reference_point' in config['triangulation'] and 'axes' in config['triangulation']:
        all_points_3d_adj, M, center = correct_coordinate_frame(config, all_points_3d, bodyparts)
    else:
        all_points_3d_adj = all_points_3d
        M = np.identity(3)
        center = np.zeros(3)

    dout = pd.DataFrame()
    for bp_num, bp in enumerate(bodyparts):
        for ax_num, axis in enumerate(['x','y','z']):
            dout[bp + '_' + axis] = all_points_3d_adj[:, bp_num, ax_num]
        dout[bp + '_error'] = all_errors[:, bp_num]
        dout[bp + '_ncams'] = num_cams[:, bp_num]
        dout[bp + '_score'] = scores_3d[:, bp_num]

    for i in range(3):
        for j in range(3):
            dout['M_{}{}'.format(i, j)] = M[i, j]

    for i in range(3):
        dout['center_{}'.format(i)] = center[i]

    dout['fnum'] = np.arange(n_frames)

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

    for name in vid_names:
        fnames = cam_videos[name]
        cam_names = [get_cam_name(config, f) for f in fnames]
        fname_dict = dict(zip(cam_names, fnames))

        output_fname = os.path.join(output_folder, name + '.csv')

        print(output_fname)
        
        if os.path.exists(output_fname):
            continue


        try:
            triangulate(config,
                        calib_folder, video_folder, pose_folder,
                        fname_dict, output_fname)
        except ValueError:
            import traceback, sys
            traceback.print_exc(file=sys.stdout)
            

triangulate_all = make_process_fun(process_session)
