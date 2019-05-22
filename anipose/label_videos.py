#!/usr/bin/env python3

import os.path
import numpy as np
from glob import glob
import pandas as pd
import cv2
import skvideo.io
from tqdm import trange

from matplotlib.pyplot import get_cmap

from .common import make_process_fun, natural_keys

def get_duration(vidname):
    metadata = skvideo.io.ffprobe(vidname)
    duration = float(metadata['video']['@duration'])
    return duration

def get_nframes(vidname):
    metadata = skvideo.io.ffprobe(vidname)
    length = int(metadata['video']['@nb_frames'])
    return length

def connect(img, points, bps, bodyparts, col=(0,255,0,255)):
    try:
        ixs = [bodyparts.index(bp) for bp in bps]
    except ValueError:
        return

    for a, b in zip(ixs, ixs[1:]):
        if np.any(np.isnan(points[[a,b]])):
            continue
        pa = tuple(np.int32(points[a]))
        pb = tuple(np.int32(points[b]))
        cv2.line(img, tuple(pa), tuple(pb), col, 4)

def connect_all(img, points, scheme, bodyparts):
    cmap = get_cmap('tab10')
    for cnum, bps in enumerate(scheme):
        col = cmap(cnum % 10, bytes=True)
        col = [int(c) for c in col]
        connect(img, points, bps, bodyparts, col)


def visualize_labels(config, labels_fname, vid_fname, outname):

    try:
        scheme = config['labeling']['scheme']
    except KeyError:
        scheme = []

    dlabs = pd.read_hdf(labels_fname)
    if len(dlabs.columns.levels) > 2:
        scorer = dlabs.columns.levels[0][0]
        dlabs = dlabs.loc[:, scorer]
    bodyparts = list(dlabs.columns.levels[0])

    cap = cv2.VideoCapture(vid_fname)
    # cap.set(1,0)

    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = skvideo.io.FFmpegWriter(outname, inputdict={
        # '-hwaccel': 'auto',
        '-framerate': str(fps),
    }, outputdict={
        '-vcodec': 'h264', '-qp': '30'
    })

    last = len(dlabs)

    cmap = get_cmap('tab10')

    points = [(dlabs[bp]['x'], dlabs[bp]['y']) for bp in bodyparts]
    points = np.array(points)

    scores = [dlabs[bp]['likelihood'] for bp in bodyparts]
    scores = np.array(scores)
    scores[np.isnan(scores)] = 0
    scores[np.isnan(points[:, 0])] = 0

    good = np.array(scores) > 0.1
    points[:, 0, :][~good] = np.nan
    points[:, 1, :][~good] = np.nan

    all_points = points

    for ix in trange(last, ncols=70):
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        labels = dlabs.iloc[ix]

        points = all_points[:, :, ix]

        connect_all(img, points, scheme, bodyparts)

        for lnum, (x, y) in enumerate(points):
            if np.isnan(x) or np.isnan(y):
                continue
            x = int(round(x))
            y = int(round(y))
            col = cmap(lnum % 10, bytes=True)
            col = [int(c) for c in col]
            cv2.circle(img,(x,y), 7, col[:3], -1)

        writer.writeFrame(img)

    cap.release()
    writer.close()



def process_session(config, session_path, filtered=False):
    pipeline_videos_raw = config['pipeline']['videos_raw']
    if filtered:
        pipeline_videos_labeled = config['pipeline']['videos_labeled_2d_filter']
        pipeline_pose = config['pipeline']['pose_2d_filter']
    else:
        pipeline_videos_labeled = config['pipeline']['videos_labeled_2d']
        pipeline_pose = config['pipeline']['pose_2d']

    print(session_path)

    labels_fnames = glob(os.path.join(session_path, pipeline_pose, '*.h5'))
    labels_fnames = sorted(labels_fnames, key=natural_keys)

    outdir = os.path.join(session_path, pipeline_videos_labeled)

    if len(labels_fnames) > 0:
        os.makedirs(outdir, exist_ok=True)

    for fname in labels_fnames:
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]

        out_fname = os.path.join(outdir, basename+'.avi')
        vidname = os.path.join(session_path, pipeline_videos_raw, basename+'.avi')

        if os.path.exists(vidname):
            if os.path.exists(out_fname) and \
               abs(get_nframes(out_fname) - get_nframes(vidname)) < 100:
                continue
            print(out_fname)

            visualize_labels(config, fname, vidname, out_fname)


label_videos_all = make_process_fun(process_session, filtered=False)
label_videos_filtered_all = make_process_fun(process_session, filtered=True)
