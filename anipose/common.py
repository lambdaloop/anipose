#!/usr/bin/env python3

import cv2
import re
import os
from collections import deque
from subprocess import check_output
import numpy as np

from aniposelib.boards import CharucoBoard, Checkerboard

def atoi(text):
    return int(text) if text.isdigit() else text

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def wc(filename):
    out = check_output(["wc", "-l", filename])
    num = out.decode('utf8').split(' ')[0]
    return int(num)

def get_data_length(fname):
    import pandas as pd
    try:
        numlines = wc(fname) - 1
    except:
        numlines = len(pd.read_csv(fname))
    return numlines

def get_video_params_cap(cap):
    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['nframes'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)
    return params

def get_video_params(fname):
    cap = cv2.VideoCapture(fname)
    params = get_video_params_cap(cap)
    cap.release()
    return params

def get_folders(path):
    folders = next(os.walk(path))[1]
    return sorted(folders)


def true_basename(fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    return basename


def get_cam_name(config, fname):
    basename = true_basename(fname)

    cam_regex = config['triangulation']['cam_regex']
    match = re.search(cam_regex, basename)

    if not match:
        return None
    else:
        name = match.groups()[0]
        return name.strip()

def get_video_name(config, fname):
    basename = true_basename(fname)

    cam_regex = config['triangulation']['cam_regex']
    vidname = re.sub(cam_regex, '', basename)
    return vidname.strip()

# TODO: get rid of skvideo dependency
def get_duration(vidname):
    import skvideo.io
    metadata = skvideo.io.ffprobe(vidname)
    duration = float(metadata['video']['@duration'])
    return duration

def get_nframes(vidname):
    import skvideo.io
    try:
        metadata = skvideo.io.ffprobe(vidname)
        length = int(metadata['video']['@nb_frames'])
        return length
    except KeyError:
        return 0

def full_path(path):
    path_user = os.path.expanduser(path)
    path_full = os.path.abspath(path_user)
    path_norm = os.path.normpath(path_full)
    return path_norm

def split_full_path(path):
    out = []
    while path != '':
        new, cur = os.path.split(path)
        if cur != '':
            out.append(cur)
        if new == path:
            out.append(new)
            break
        path = new
    return list(reversed(out))


def process_all(config, process_session, **args):

    pipeline_prefix = config['path']
    nesting = config['nesting']

    output = dict()

    if nesting == 0:
        output[()] = process_session(config, pipeline_prefix, **args)
        return output

    folders = get_folders(pipeline_prefix)
    level = 1

    q = deque()

    next_folders = [ (os.path.join(pipeline_prefix, folder),
                      (folder,),
                      level)
                     for folder in folders ]
    q.extend(next_folders)

    while len(q) != 0:
        path, past_folders, level = q.pop()

        if nesting < 0:
            output[past_folders] = process_session(config, path, **args)

            folders = get_folders(path)
            next_folders = [ (os.path.join(path, folder),
                              past_folders + (folder,),
                              level+1)
                             for folder in folders ]
            q.extend(next_folders)
        else:
            if level == nesting:
                output[past_folders] = process_session(config, path, **args)
            elif level > nesting:
                continue
            elif level < nesting:
                folders = get_folders(path)
                next_folders = [ (os.path.join(path, folder),
                                  past_folders + (folder,),
                                  level+1)
                                 for folder in folders ]
                q.extend(next_folders)

    return output

def make_process_fun(process_session, **args):
    def fun(config):
        return process_all(config, process_session, **args)
    return fun

def find_calibration_folder(config, session_path):
    pipeline_calibration_videos = config['pipeline']['calibration_videos']
    nesting = config['nesting']

    # TODO: fix this for nesting = -1
    level = nesting
    curpath = session_path

    while level >= 0:
        checkpath = os.path.join(curpath, pipeline_calibration_videos)
        if os.path.isdir(checkpath):
            return curpath

        curpath = os.path.dirname(curpath)
        level -= 1



def get_calibration_board(config):

    calib = config['calibration']
    board_size = calib['board_size']
    board_type = calib['board_type'].lower()

    manual_verification = config['manual_verification']
    manually_verify = manual_verification['manually_verify']

    if board_type == 'aruco':
        raise NotImplementedError("aruco board is not implemented with the current pipeline")
    elif board_type == 'charuco':
        board = CharucoBoard(
            board_size[0], board_size[1],
            calib['board_square_side_length'],
            calib['board_marker_length'],
            calib['board_marker_bits'],
            calib['board_marker_dict_number'],
            manually_verify=manually_verify)



    elif board_type == 'checkerboard':
        board = Checkerboard(board_size[0], board_size[1],
                             calib['board_square_side_length'], manually_verify=manually_verify)
    else:
        raise ValueError("board_type should be one of "
                         "'aruco', 'charuco', or 'checkerboard' not '{}'".format(
                             board_type))

    return board


## TODO: support checkerboard drawing
def get_calibration_board_image(config):
    board = get_calibration_board(config)
    numx, numy = board.get_size()
    size = numx*200, numy*200
    img = board.draw(size)
    return img
