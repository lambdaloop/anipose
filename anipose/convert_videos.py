#!/usr/bin/env python3

import os
import os.path
import subprocess
from glob import glob
import sys
from collections import deque
import re
import cv2
from multiprocessing import Pool
from .common import process_all, get_video_params, natural_keys

if len(sys.argv) < 2:
    source_dir = os.getcwd()
else:
    source_dir = sys.argv[1]

source_dir = os.path.abspath(source_dir)


def same_length(vid1, vid2):
    params1 = get_video_params(vid1)
    params2 = get_video_params(vid2)
    return abs(params1['nframes'] - params2['nframes']) < 5


def process_video(fname, outname, video_speed):
    # print(outname, 'started')
    if os.path.exists(outname) and same_length(vidname, outname):
        return

    params = get_video_params(fname)
    params['fps']

    if video_speed != 1:
        vfilter = 'setpts={:.2f}*PTS, fps=fps={:.2f}, pad=ceil(iw/2)*2:ceil(ih/2)*2'.format(
            1.0/video_speed, params['fps']*video_speed
        )
    else:
        vfilter = 'pad=ceil(iw/2)*2:ceil(ih/2)*2'

    print(outname)
    subprocess.run(['ffmpeg', '-y',
                    '-i', fname,
                    '-hide_banner', '-loglevel', 'error', # '-stats',
                    '-vcodec', 'h264', '-qp', '28', '-pix_fmt', 'yuv420p',
                    '-filter:v', vfilter,
                    outname])
    # print(outname, 'finished')

def process_folder(config, path):
    print(path)

    vidnames = glob(os.path.join(path, config['pipeline']['videos_raw'],
                                 '*.' + config['video_extension']))
    vidnames = sorted(vidnames, key=natural_keys)

    outpath = os.path.join(path, config['pipeline']['videos_raw_mp4'])

    os.makedirs(outpath, exist_ok=True)

    pool = Pool(3)

    for vidname in vidnames:
        basename = os.path.basename(vidname)
        base, ext = os.path.splitext(basename)
        outname = os.path.join(outpath, base+'.mp4')
        # if not (os.path.exists(outname) and same_length(vidname, outname)):
            # process_video(vidname, outname)
        pool.apply_async(process_video, (vidname, outname, config['converted_video_speed']))

    pool.close()
    pool.join()

    return vidnames

def convert_all(config):

    process_all(config, process_folder)
