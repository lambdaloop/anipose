#!/usr/bin/env python3

from tqdm import tqdm, trange
import numpy as np
from collections import defaultdict
import os.path, os
import numpy as np
import pickle
import pandas as pd
import toml
from numpy import array as arr
from glob import glob


## hack for hdf5 for testing
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'


def load_intrinsics(folder, cam_names):
    intrinsics = {}
    for cname in cam_names:
        fname = os.path.join(folder, 'intrinsics_{}.toml'.format(cname))
        intrinsics[cname] = toml.load(fname)
    return intrinsics

def load_extrinsics(folder):
    extrinsics = toml.load(os.path.join(folder, 'extrinsics.toml'))
    extrinsics_out = dict()
    for k, v in extrinsics.items():
        new_k = tuple(k.split('_'))
        extrinsics_out[new_k] = v
    return extrinsics_out

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

def minmax(pts, p=0):
    good = ~np.isnan(pts)
    xs = pts[good]
    return np.percentile(xs, [p, 100-p])

def rerange(px, highrange):
    low, high = px
    mid = (low+high)/2
    return [mid - highrange/2, mid + highrange/2]



def triangulate(calib_folder, video_folder, pose_folder,
                fname_dict, output_fname, cam_align='C'):

    # video_format = os.path.join(video_folder, vidname + '_{}.avi')
    # pose_format = os.path.join(pose_folder, vidname + '_{}.h5')

    record_fname = os.path.join(video_folder, 'recorder.toml')

    if not os.path.exists(record_fname):
        return

    record_dict = toml.load(record_fname)

    
    cam_names, pose_names = list(zip(*sorted(fname_dict.items())))

    intrinsics = load_intrinsics(calib_folder, cam_names)
    extrinsics = load_extrinsics(calib_folder)

    offsets = []
    cam_mats = []
    for cname in cam_names:
        left = expand_matrix(arr(intrinsics[cname]['camera_mat']))
        if cname == cam_align:
            right = np.identity(4)
        else:
            right = arr(extrinsics[(cname, cam_align)])
        mat = np.matmul(left, right)
        cam_mats.append(mat)
        offset = record_dict['cameras'][cname]['video']['ROIPosition']
        offsets.append(offset)


    maxlen = 0
    caps = dict()
    for pose_name in pose_names:
        dd =  pd.read_hdf(pose_name)
        length = len(dd.index)
        maxlen = max(maxlen, length)
 
    dd =  pd.read_hdf(pose_names[0])
    bodyparts = arr(dd.columns.levels[0])

    # frame, camera, bodypart, xy
    all_points_raw = np.zeros((length, len(cam_names), len(bodyparts), 2))
    all_scores = np.zeros((length, len(cam_names), len(bodyparts)))

    for ix_cam, (cam_name, pose_name, offset) in \
        enumerate(zip(cam_names, pose_names, offsets)):
        dd = pd.read_hdf(pose_name)
        index = arr(dd.index)
        for ix_bp, bp in enumerate(bodyparts):
            X = arr(dd[bp])
            all_points_raw[index, ix_cam, ix_bp, :] = X[:, :2] + [offset[0], offset[1]]
            all_scores[index, ix_cam, ix_bp] = X[:, 2]


    shape = all_points_raw.shape

    all_points_3d = np.zeros((shape[0], shape[2], 3))
    all_points_3d.fill(np.nan)

    errors = np.zeros((shape[0], shape[2]))
    errors.fill(np.nan)

    for i in trange(all_points_raw.shape[0], ncols=70):
        for j in range(all_points_raw.shape[2]):
            pts = all_points_raw[i, :, j, :]
            if ~np.any(np.isnan(pts)):
                p3d = triangulate_simple(pts, cam_mats)
                all_points_3d[i, j] = p3d[:3]
                errors[i,j] = reprojection_error(p3d, pts, cam_mats)


    dout = pd.DataFrame()
    for bp_num, bp in enumerate(bodyparts):
        for ax_num, axis in enumerate(['x','y','z']):
            dout[bp + '_' + axis] = all_points_3d[:, bp_num, ax_num]
        dout[bp + '_error'] = errors[:, bp_num]

    dout['fnum'] = np.arange(length)
        
    dout.to_csv(output_fname, index=False)


def process_session(config, session_path):
    pipeline_videos_raw = config['pipeline_videos_raw']
    pipeline_calibration_results = config['pipeline_calibration_results']
    pipeline_pose = config['pipeline_pose_2d']
    pipeline_3d = config['pipeline_pose_3d']

    
    calibration_path = find_calibration_folder(config, session_path)
    if calibration_path is None:
        return
    
    calib_folder = os.path.join(calibration_path, pipeline_calibration)
    video_folder = os.path.join(session_path, pipeline_videos_raw)
    pose_folder = os.path.join(session_path, pipeline_pose)
    output_folder = os.path.join(session_path, pipeline_3d)

    os.makedirs(output_folder, exist_ok=True)

    pose_files = glob(os.path.join(pose_folder, '*.h5'))

    cam_videos = defaultdict(list)

    for pf in pose_files:
        name = get_video_name(pf)
        cam_videos[name].append(pf)

    vid_names = cam_videos.keys()
    vid_names = sorted(vid_names)

    fname_dicts = []
    for name in vid_names:
        print(name)
        fnames = cam_videos[name]
        cam_names = [get_cam_name(f) for f in fnames]
        fname_dict = dict(zip(cam_names, fnames))
        fname_dicts.append(fname_dict)

        output_fname = os.path.join(output_folder, name + '.csv')

        if os.path.exists(output_fname):
            continue

        triangulate(calib_folder, video_folder, pose_folder,
                    fname_dict, output_fname)


triangulate_all = make_process_fun(process_session)
