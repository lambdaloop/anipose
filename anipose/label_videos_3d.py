#!/usr/bin/env python3

from mayavi import mlab
mlab.options.offscreen = True

import numpy as np
from glob import glob
import pandas as pd
import os.path
import cv2
import sys
import skvideo.io
from tqdm import tqdm, trange
import sys
from collections import defaultdict
from matplotlib.pyplot import get_cmap

from .common import make_process_fun, get_nframes, get_video_name, get_video_params, get_data_length, natural_keys

def connect(points, bps, bp_dict, color):
    ixs = [bp_dict[bp] for bp in bps]
    return mlab.plot3d(points[ixs, 0], -points[ixs, 1], points[ixs, 2],
                       np.ones(len(ixs)), reset_zoom=False,
                       color=color, tube_radius=None, line_width=8)

def connect_all(points, scheme, bp_dict, cmap):
    lines = []
    for i, bps in enumerate(scheme):
        line = connect(points, bps, bp_dict, color=cmap(i)[:3])
        lines.append(line)
    return lines

def update_line(line, points, bps, bp_dict):
    ixs = [bp_dict[bp] for bp in bps]
    # ixs = [bodyparts.index(bp) for bp in bps]
    new = np.vstack([points[ixs, 0], -points[ixs, 1], points[ixs, 2]]).T
    line.mlab_source.points = new

def update_all_lines(lines, points, scheme, bp_dict):
    for line, bps in zip(lines, scheme):
        update_line(line, points, bps, bp_dict)

def get_points(dx, bodyparts):
    points = [(dx[bp+'_x'], dx[bp+'_y'], dx[bp+'_z']) for bp in bodyparts]
    # scores = [dx[bp+'_score'] for bp in bodyparts]
    errors = np.array([dx[bp+'_error'] for bp in bodyparts])
    ncams = np.array([dx[bp+'_ncams'] for bp in bodyparts])
    # good = (np.array(scores) > 0.1) & (np.array(errors) < 35)

    ## TODO: add checking on scores here
    ## TODO: make error thresholds configurable
    errors[np.isnan(errors)] = 10000
    ncams[np.isnan(ncams)] = 0
    good = (errors < 250) #&  (ncams >= 3)

    points = np.array(points)
    points[~good] = np.nan

    return points



def visualize_labels(config, labels_fname, outname, fps=300):

    try:
        scheme = config['labeling']['scheme']
    except KeyError:
        scheme = []

    data = pd.read_csv(labels_fname)
    cols = [x for x in data.columns if '_error' in x]

    if len(scheme) == 0:
        bodyparts = [c.replace('_error', '') for c in cols]
    else:
        bodyparts = sorted(set([x for dx in scheme for x in dx]))

    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))

    nparts = len(bodyparts)
    framenums = np.array(data['fnum'])
    framedict = dict(zip(data['fnum'], data.index))

    writer = skvideo.io.FFmpegWriter(outname, inputdict={
        # '-hwaccel': 'auto',
        '-framerate': str(fps),
    }, outputdict={
        '-vcodec': 'h264', '-qp': '30'
    })

    cmap = get_cmap('tab10')


    dx = data.iloc[20]
    points = get_points(dx, bodyparts)

    s = np.arange(points.shape[0])
    good = ~np.isnan(points[:, 0])

    fig = mlab.figure(bgcolor=(1,1,1), size=(500,500))
    fig.scene.anti_aliasing_frames = 2

    ## TODO: estimate scale_factor from the spead of points
    mlab.clf()
    pts = mlab.points3d(points[:, 0], -points[:, 1], points[:, 2], s,
                        scale_mode='none', scale_factor=0.05)
    lines = connect_all(points, scheme, bp_dict, cmap)
    mlab.orientation_axes()

    view = list(mlab.view())

    for framenum in trange(data.shape[0],ncols=70):
        fig.scene.disable_render = True

        if framenum in framedict:
            ix = framedict[framenum]
            dx = data.iloc[ix]
            points = get_points(dx, bodyparts)
        else:
            points = np.ones((nparts, 3))*np.nan

        s = np.arange(points.shape[0])
        good = ~np.isnan(points[:, 0])

        new = np.vstack([points[:, 0], -points[:, 1], points[:, 2]]).T
        pts.mlab_source.points = new
        update_all_lines(lines, points, scheme, bp_dict)

        fig.scene.disable_render = False

        img = mlab.screenshot()

        view[0] += -0.2
        mlab.view(*view, reset_roll=False)

        writer.writeFrame(img)

    mlab.close(all=True)
    writer.close()



def process_session(config, session_path):
    pipeline_videos_raw = config['pipeline']['videos_raw']
    pipeline_videos_labeled_3d = config['pipeline']['videos_labeled_3d']
    pipeline_3d = config['pipeline']['pose_3d']


    vid_fnames = glob(os.path.join(session_path,
                                   pipeline_videos_raw, "*.avi"))
    orig_fnames = defaultdict(list)
    for vid in vid_fnames:
        vidname = get_video_name(config, vid)
        orig_fnames[vidname].append(vid)

    labels_fnames = glob(os.path.join(session_path,
                                      pipeline_3d, '*.csv'))
    labels_fnames = sorted(labels_fnames, key=natural_keys)


    outdir = os.path.join(session_path, pipeline_videos_labeled_3d)

    if len(labels_fnames) > 0:
        os.makedirs(outdir, exist_ok=True)

    for fname in labels_fnames:
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]

        out_fname = os.path.join(outdir, basename+'.avi')

        if os.path.exists(out_fname) and \
           abs(get_nframes(out_fname) - get_data_length(fname)) < 100:
            continue
        print(out_fname)

        some_vid = orig_fnames[basename][0]
        params = get_video_params(some_vid)

        visualize_labels(config, fname, out_fname, params['fps'])


label_videos_3d_all = make_process_fun(process_session)
