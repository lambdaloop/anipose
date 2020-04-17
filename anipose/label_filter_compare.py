#!/usr/bin/env python3

import numpy as np
from glob import glob
import pandas as pd
import os.path
import cv2
from tqdm import tqdm, trange
from collections import defaultdict
from scipy import signal
import queue
import threading

from aniposelib.cameras import CameraGroup

from .common import make_process_fun, get_nframes, \
    get_video_name, get_cam_name, \
    get_video_params, get_video_params_cap, \
    get_data_length, natural_keys, true_basename, find_calibration_folder, \
    nan_helper

from .triangulate import load_offsets_dict

from .label_videos import label_frame


def write_frame_thread(writer, q):
    while True:
        frame = q.get(block=True)
        if frame is None:
            return
        writer.write(frame)
        # writer.writeFrame(frame)

def turn_to_black(frame):
    frame = np.float32(frame)
    white = np.all(frame > 220, axis=2)
    frame[white] = [0,0,0]
    frame[~white] *= 1.5
    frame = np.clip(frame, 0, 255).astype('uint8')
    return frame


def read_frames(caps_2d):
    frames_2d = []
    for cap in caps_2d:
        ret, frame = cap.read()
        if not ret:
            return False, None
        frames_2d.append(frame)

    return ret, frames_2d


def get_plotting_params(caps_2d):
    params_2d = [get_video_params_cap(c) for c in caps_2d]
    height_2d = max([p['height'] for p in params_2d])
    widths_2d = [round(p['width'] * height_2d/p['height']) for p in params_2d]

    width_total = sum(widths_2d)
    height_total = height_2d * 3

    nframes = min([p['nframes'] for p in params_2d])

    fps = np.mean([p['fps'] for p in params_2d])

    d = {
        'height_2d': height_2d,
        'widths_2d': widths_2d,
        'nframes': nframes,
        'fps': fps,
        'width_total': width_total,
        'height_total': height_total,
    }

    return d

def get_start_image(pp):
    start_img = np.zeros((pp['height_total'], pp['width_total'], 3), dtype='uint8')
    start_img[:] = 0

    return start_img


def draw_data(start_img, frames_raw, frames_2d, frames_2d_filt, pp):
    height_2d = pp['height_2d']
    widths_2d = pp['widths_2d']

    width_total = pp['width_total']
    height_total = pp['height_total']

    frames_raw_resized = [cv2.resize(f, (w, height_2d))
                          for f, w in zip(frames_raw, widths_2d)]
    
    frames_2d_resized = [cv2.resize(f, (w, height_2d))
                         for f, w in zip(frames_2d, widths_2d)]

    frames_2d_filt_resized = [cv2.resize(f, (w, height_2d))
                              for f, w in zip(frames_2d_filt, widths_2d)]
    
    imout = np.copy(start_img)
    imout[0:height_2d] = np.hstack(frames_raw_resized)
    imout[height_2d:height_2d*2] = np.hstack(frames_2d_resized)
    imout[height_2d*2:height_2d*3] = np.hstack(frames_2d_filt_resized)

    return imout


def visualize_compare(config, fnames_raw, fnames_2d, fnames_2d_filt, out_fname):

    caps_raw = [cv2.VideoCapture(v) for v in fnames_raw]
    caps_2d = [cv2.VideoCapture(v) for v in fnames_2d]
    caps_2d_filt = [cv2.VideoCapture(v) for v in fnames_2d_filt]

    pp = get_plotting_params(caps_2d)
    nframes = pp['nframes']
    fps = pp['fps']
    start_img = get_start_image(pp)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_fname, fourcc, round(fps, ndigits=2),
                             (pp['width_total'], pp['height_total']))

    q = queue.Queue(maxsize=50)

    thread = threading.Thread(target=write_frame_thread,
                              args=(writer, q))
    thread.start()

    for framenum in trange(nframes, ncols=70):
        ret0, frames_raw = read_frames(caps_raw)
        ret1, frames_2d = read_frames(caps_2d)
        ret2, frames_2d_filt = read_frames(caps_2d_filt)
        if not all([ret0, ret1, ret2]):
            break

        imout = draw_data(start_img, frames_raw, frames_2d, frames_2d_filt, pp)
        q.put(imout)

    for cap in caps_raw:
        cap.release()
    for cap in caps_2d:
        cap.release()
    for cap in caps_2d_filt:
        cap.release()

    q.put(None)
    thread.join()
    writer.release()

def process_session(config, session_path):
    pipeline_videos_labeled_2d = config['pipeline']['videos_labeled_2d']
    pipeline_videos_labeled_2d_filter = config['pipeline']['videos_labeled_2d_filter']
    pipeline_videos_raw = config['pipeline']['videos_raw']
    pipeline_videos_compare = config['pipeline']['videos_compare']

    video_ext = config['video_extension']

    vid_fnames = glob(os.path.join(session_path, pipeline_videos_raw, "*." + video_ext))

    fnames_raw = defaultdict(list)
    for vid in vid_fnames:
        vidname = get_video_name(config, vid)
        fnames_raw[vidname].append(vid)

    outdir = os.path.join(session_path, pipeline_videos_compare)

    if len(vid_fnames) > 0:
        os.makedirs(outdir, exist_ok=True)

    for vidname in sorted(fnames_raw.keys(), key=natural_keys):
        out_fname = os.path.join(outdir, vidname+'.mp4')
        
        vids_raw = sorted(fnames_raw[vidname], key=natural_keys)
        vid_fname = vids_raw[0]
        
        if os.path.exists(out_fname) and \
           abs(get_nframes(out_fname) - get_nframes(vid_fname)) < 100:
            continue

        vids_2d = [os.path.join(session_path, pipeline_videos_labeled_2d,
                                true_basename(f) + '.mp4')
                   for f in vids_raw]

        vids_2d_filtered = [os.path.join(session_path,
                                         pipeline_videos_labeled_2d_filter,
                                         true_basename(f) + '.mp4')
                            for f in vids_raw]
        
        if not all([os.path.exists(f) for f in vids_2d]):
            print(out_fname, 'missing labeled 2d videos')
            continue

        if not all([os.path.exists(f) for f in vids_2d_filtered]):
            print(out_fname, 'missing labeled filtered 2d videos')
            continue

        print(out_fname)
        
        visualize_compare(config, vids_raw, vids_2d, vids_2d_filtered, out_fname)


label_filter_compare_all = make_process_fun(process_session)
