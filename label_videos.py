#!/usr/bin/env python3

import os.path

import numpy as np
from glob import glob
import pandas as pd
import os.path
import cv2
import skvideo.io
from tqdm import tqdm, trange

from matplotlib.pyplot import get_cmap

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


# scheme = [
#     ['body-coxa-right', 'coxa-femur-right', 'femur-tibia-right',
#      'tibia-tarsus-right', 'tarsus-end-right'],
#     ['body-coxa-left', 'coxa-femur-left', 'femur-tibia-left',
#      'tibia-tarsus-left', 'tarsus-end-left']
# ]

# TODO: read label scheme from config file
scheme = [
    ['L1A', 'L1B', 'L1C', 'L1D', 'L1E'],
    ['L2A', 'L2B', 'L2C', 'L2D', 'L2E'],
    ['L3A', 'L3B', 'L3C', 'L3D', 'L3E']
]


def visualize_labels(labels_fname, vid_fname, outname):

    dlabs = pd.read_hdf(labels_fname)
    if len(dlabs.columns.levels) > 2:
        scorer = dlabs.columns.levels[0][0]
        dlabs = dlabs.loc[:, scorer]
    bodyparts = list(dlabs.columns.levels[0])

    cap = cv2.VideoCapture(vid_fname)
    # cap.set(1,0)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = skvideo.io.FFmpegWriter(outname, inputdict={
        # '-hwaccel': 'auto',
        '-framerate': str(fps),
    }, outputdict={
        '-vcodec': 'h264', '-qp': '30'
    })

    last = len(dlabs)

    cmap = get_cmap('tab10')


    for ix in trange(last, ncols=70):
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if True:
            labels = dlabs.iloc[ix]

            points = [(labels[bp]['x'], labels[bp]['y']) for bp in bodyparts]
            points = np.array(points)

            scores = [labels[bp]['likelihood'] for bp in bodyparts]
            scores = np.array(scores)
            scores[np.isnan(scores)] = 0
            scores[np.isnan(points[:, 0])] = 0
            good = np.array(scores) > 0.1

            points[~good] = np.nan

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


def get_folders(path):
    folders = next(os.walk(path))[1]
    return sorted(folders)


def process_session(config, session_path):
    pipeline_videos_raw = config['pipeline_videos_raw']
    pipeline_videos_labeled = config['pipeline_videos_labeled_2d']
    pipeline_pose = config['pipeline_pose_2d']

    labels_fnames = glob(os.path.join(session_path, pipeline_pose, '*.h5'))
    labels_fnames = sorted(labels_fnames)

    outdir = os.path.join(session_path, pipeline_videos_labeled)
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

            visualize_labels(fname, vidname, out_fname)


def label_videos_all(config):
    pipeline_prefix = config['path']

    sessions = get_folders(pipeline_prefix)

    for session in sessions:
        print(session)
        session_path = os.path.join(pipeline_prefix, session)
        process_session(config, session_path)
