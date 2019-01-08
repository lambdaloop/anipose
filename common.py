import cv2
import re
import os, os.path
from collections import deque

def get_video_params(fname):
    cap = cv2.VideoCapture(fname)

    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return params

def get_folders(path):
    folders = next(os.walk(path))[1]
    return sorted(folders)

def get_cam_name(config, fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]

    cam_regex = config['cam_regex']
    match = re.search(cam_regex, basename)

    if not match:
        return None
    else:
        return match.groups()[0]

def get_video_name(config, fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]

    cam_regex = config['cam_regex']
    return re.sub(cam_regex, '', basename)

def get_duration(vidname):
    metadata = skvideo.io.ffprobe(vidname)
    duration = float(metadata['video']['@duration'])
    return duration

def get_nframes(vidname):
    metadata = skvideo.io.ffprobe(vidname)
    length = int(metadata['video']['@nb_frames'])
    return length

def process_all(config, process_session, *args):
    pipeline_prefix = config['path']
    nesting = config['nesting']

    if nesting == 0:
        process_session(config, pipeline_prefix, *args)
        return
    
    folders = get_folders(pipeline_prefix)
    level = 1

    q = deque()
    
    next_folders = [ (os.path.join(pipeline_prefix, folder), level)
                     for folder in folders ]
    q.extend(next_folders)

    while len(q) != 0:
        path, level = q.pop()
        if level == nesting:
            process_session(config, path, *args)
        elif level > nesting:
            continue
        elif level < nesting:
            folders = get_folders(path)
            next_folders = [ (os.path.join(path, folder), level+1)
                             for folder in folders ]
            q.extend(next_folders)
    
def make_process_fun(process_session):
    def fun(config):
        return process_all(config, process_session)
    return fun
