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

from .common import make_process_fun, get_nframes, get_video_name, \
    get_video_params, get_video_params_cap, \
    get_data_length, natural_keys, true_basename

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

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


def read_frames(caps_2d, cap_3d):
    frames_2d = []
    for cap in caps_2d:
        ret, frame = cap.read()
        if not ret:
            return False, None, None
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame
        frames_2d.append(img)

    ret, frame = cap_3d.read()
    if not ret:
        return False, None, None
    frame_3d = frame
    # frame_3d = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_3d = turn_to_black(frame_3d)

    return ret, frames_2d, frame_3d

def get_video_params_cap(cap):
    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['nframes'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)
    return params

def draw_seq(img, seq, rect, range_y=(None, None),
             color=(0, 0, 0), thickness=5):
    low, high = range_y

    if low is None:
        low = np.min(seq)
    if high is None:
        high = np.max(seq)
    s = np.clip(seq, low, high)
    s = (s - low)/(high-low)

    left, right, top, bottom = rect
    height = bottom - top

    xs = np.linspace(left, right, num=len(seq))
    ys = (1-s) * height + top
    pointlist = list(zip(xs, ys))
    pointlist = [(x, y) for x, y in pointlist if not np.isnan(y)]
    pointlist = np.int32([pointlist])

    cv2.polylines(img, pointlist, False, color,
                  thickness=thickness, lineType=cv2.LINE_AA)

def mapto(x, fromLow, fromHigh, toLow, toHigh):
    norm = (x - fromLow) / (fromHigh-fromLow)
    return norm * (toHigh - toLow) + toLow

def draw_axis_y(img, rect, range_y, label,
                num_ticks=5,
                color=(0, 0, 0), thickness=5):
    left, right, top, bottom = rect
    height = bottom - top

    left_start = left - 10

    low, high = range_y
    ticks = np.linspace(low, high, num_ticks+2)[1:-1]

    cv2.line(img, (left_start, top+10), (left_start, bottom-10),
             color, thickness=thickness)

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cv2.getFontScaleFromHeight(
        font_face, int(round(height / (num_ticks + 3))))

    for t in ticks:
        y = mapto(t, low, high, bottom, top)
        y = int(round(y))
        lab = str(int(round(t)))
        (w, h), baseline = cv2.getTextSize(lab, font_face, font_scale, 2)
        cv2.line(img, (left_start-5, y), (left_start, y),
                 color, thickness=thickness)
        cv2.putText(img, lab, (left_start-30-w//2, y + baseline),
                    font_face, 0.9, color, thickness=2)


    imgnew = np.zeros((img.shape[1], img.shape[0]), dtype='uint8')
    cv2.putText(imgnew, label, (img.shape[0] - top - 130, left-100), font_face, font_scale*0.7, 255, thickness=2)
    imgnew = cv2.rotate(imgnew, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img[imgnew > 0] = color

    return img


def get_plotting_params(caps_2d, cap_3d, ang_names):

    height_angle = 175
    spacing_angle = 40
    spacing_videos = 20

    n_angles = len(ang_names)

    params_2d = [get_video_params_cap(c) for c in caps_2d]
    height_2d = max([p['height'] for p in params_2d])
    widths_2d = [round(p['width'] * height_2d/p['height']) for p in params_2d]

    param_3d = get_video_params_cap(cap_3d)
    height_3d = param_3d['height']
    width_3d = param_3d['width']

    start_3d = height_2d + spacing_videos
    start_angles = start_3d + height_3d + spacing_videos + spacing_angle

    width_total = sum(widths_2d)
    height_total = height_2d + spacing_videos + height_3d + spacing_videos + \
        height_angle * n_angles + spacing_angle*(n_angles+1)

    nframes = min([p['nframes'] for p in params_2d])
    nframes = min(nframes, param_3d['nframes'])

    fps = param_3d['fps']

    height_font = spacing_angle//2
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cv2.getFontScaleFromHeight(
        font_face, height_font)
    font_scale = int(round(font_scale))
    font_color = (0, 0, 0)
    font_thickness = 2

    mid_3d = int((width_total - width_3d) / 2)

    d = {
        'height_angle': height_angle,
        'spacing_angle': spacing_angle,
        'spacing_videos': spacing_videos,

        'height_2d': height_2d,
        'widths_2d': widths_2d,
        'start_3d': start_3d,
        'width_3d': width_3d,
        'mid_3d': mid_3d,
        'height_3d': height_3d,

        'nframes': nframes,
        'fps': fps,

        'width_total': width_total,
        'height_total': height_total,

        'start_3d': start_3d,
        'start_angles': start_angles,

        'height_font': height_font,
        'font_face': font_face,
        'font_scale': font_scale,
        'font_color': font_color,
        'font_thickness': font_thickness,
    }

    return d

def get_start_image(pp, ang_names):

    start_img = np.zeros((pp['height_total'], pp['width_total'], 3), dtype='uint8')
    start_img[:] = 255

    for angnum, name in enumerate(ang_names):
        start_y = pp['start_angles'] + (pp['height_angle'] + pp['spacing_angle'])*angnum
        rect = (150, pp['width_total']-100, start_y, start_y + pp['height_angle'])

        font_size, baseline = cv2.getTextSize(
            name, pp['font_face'], pp['font_scale'], pp['font_thickness'])
        fw, fh = font_size

        text_xy = (pp['width_total'] // 2 - fw // 2, start_y)
        cv2.putText(start_img, name, text_xy, pp['font_face'], pp['font_scale'], pp['font_color'],
                    thickness=2, lineType=cv2.LINE_AA)

        draw_axis_y(start_img, rect, (0, 180), 'Angle',
                    num_ticks=3, thickness=4)

    return start_img


def draw_data(start_img, frames_2d, frame_3d, all_angles, pp):
    height_2d = pp['height_2d']
    widths_2d = pp['widths_2d']

    start_3d = pp['start_3d']
    height_3d = pp['height_3d']
    width_3d = pp['width_3d']
    mid_3d = pp['mid_3d']

    start_angles = pp['start_angles']
    height_angle = pp['height_angle']
    spacing_angle = pp['spacing_angle']

    width_total = pp['width_total']
    height_total = pp['height_total']

    frames_2d_resized = [cv2.resize(f, (w, height_2d))
                         for f, w in zip(frames_2d, widths_2d)]

    imout = np.copy(start_img)
    imout[0:height_2d] = np.hstack(frames_2d_resized)
    imout[start_3d:(start_3d + height_3d), mid_3d:(mid_3d+width_3d)] = frame_3d

    data_color = (0,0,0)
    indicator_color = (150,150,150)

    for angnum, angles in enumerate(all_angles):
        start_y = start_angles + (height_angle + spacing_angle)*angnum
        rect = (150, width_total-100, start_y, start_y + height_angle)
        left, right, top, bottom = rect

        draw_seq(imout, angles, rect,
                 range_y=(0, 180), color=data_color, thickness=2)
        x = (left+right)//2
        cv2.line(imout, (x, top+15), (x, bottom-15),
                 indicator_color, thickness=2)

    return imout


def visualize_combined(config, angle_fname, fnames_2d, fname_3d, out_fname):
    angles = pd.read_csv(angle_fname)

    bad_cols = ['fnum']
    ang_names = [col for col in angles.columns if col not in bad_cols]

    ang_values = dict()

    for name in ang_names:
        vals = np.array(angles[name])
        angf = signal.medfilt(vals, kernel_size=5)
        err = np.abs(angf - vals)
        err[np.isnan(err)] = 10000

        vals[err > 10] = np.nan
        nans, ix = nan_helper(vals)
        # some data missing, but not too much
        if np.sum(nans) > 0 and np.sum(~nans) > 5:
            vals[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])

        ang_values[name] = vals

    caps_2d = [cv2.VideoCapture(v) for v in fnames_2d]
    cap_3d = cv2.VideoCapture(fname_3d)

    pp = get_plotting_params(caps_2d, cap_3d, ang_names)
    nframes = pp['nframes']
    fps = pp['fps']
    start_img = get_start_image(pp, ang_names)

    ang_window_size = 100
    pad_size = ang_window_size

    ang_values_padded = dict()
    for name, angles in ang_values.items():
        ang_values_padded[name] = np.pad(angles, pad_size,
                                         mode='constant', constant_values=np.nan)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_fname, fourcc, fps,
                             (pp['width_total'], pp['height_total']))

    q = queue.Queue()

    thread = threading.Thread(target=write_frame_thread,
                              args=(writer, q))
    thread.start()

    for framenum in trange(nframes, ncols=70):
        ret, frames_2d, frame_3d = read_frames(caps_2d, cap_3d)
        if not ret:
            break

        all_angles = []
        for angnum, name in enumerate(ang_names):
            a = framenum + pad_size - ang_window_size//2
            b = a + ang_window_size
            angles = ang_values_padded[name][a:b]
            all_angles.append(angles)

        imout = draw_data(start_img, frames_2d, frame_3d, all_angles, pp)
        q.put_nowait(imout)

    q.put_nowait(None)
    thread.join()
    writer.release()

def process_session(config, session_path):
    filtered = config['filter']['enabled']
    if filtered:
        pipeline_videos_labeled_2d = config['pipeline']['videos_labeled_2d_filter']
    else:
        pipeline_videos_labeled_2d = config['pipeline']['videos_labeled_2d']
    pipeline_videos_labeled_3d = config['pipeline']['videos_labeled_3d']
    pipeline_angles = config['pipeline']['angles']
    pipeline_videos_combined = config['pipeline']['videos_combined']

    vid_fnames_2d = glob(os.path.join(session_path,
                                      pipeline_videos_labeled_2d, "*.avi"))

    vid_fnames_3d = glob(os.path.join(session_path,
                                      pipeline_videos_labeled_3d, "*.avi"))

    fnames_2d = defaultdict(list)
    for vid in vid_fnames_2d:
        vidname = get_video_name(config, vid)
        fnames_2d[vidname].append(vid)

    fnames_3d = defaultdict(list)
    for vid in vid_fnames_3d:
        vidname = true_basename(vid)
        fnames_3d[vidname].append(vid)

    angle_fnames = glob(os.path.join(session_path,
                                     pipeline_angles, '*.csv'))
    angle_fnames = sorted(angle_fnames, key=natural_keys)

    outdir = os.path.join(session_path, pipeline_videos_combined)

    if len(angle_fnames) > 0:
        os.makedirs(outdir, exist_ok=True)

    for angle_fname in angle_fnames:
        basename = true_basename(angle_fname)

        out_fname = os.path.join(outdir, basename+'.avi')

        if os.path.exists(out_fname) and \
           abs(get_nframes(out_fname) - get_data_length(angle_fname)) < 100:
            continue

        if len(fnames_2d[basename]) == 0:
            print(out_fname, 'missing 2d')
            continue

        if len(fnames_3d[basename]) == 0:
            print(out_fname, 'missing 3d')
            continue

        fname_3d_current = fnames_3d[basename][0]
        fnames_2d_current = fnames_2d[basename]

        print(out_fname)

        visualize_combined(config, angle_fname,
                           fnames_2d_current, fname_3d_current, out_fname)


label_combined_all = make_process_fun(process_session)
