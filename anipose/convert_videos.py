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


def process_video(fname, outname, encoding_params):
    # print(outname, 'started')
    if os.path.exists(outname) and same_length(fname, outname):
        return

    video_speed = encoding_params.get('converted_video_speed', 1)
    quality = encoding_params.get('video_quality', 28)
    gpu_enabled = encoding_params.get('gpu_enabled', False)

    params = get_video_params(fname)

    if video_speed != 1:
        vfilter = 'setpts={:.2f}*PTS, fps=fps={:.2f}, pad=ceil(iw/2)*2:ceil(ih/2)*2'.format(
            1.0/video_speed, params['fps']*video_speed
        )
    else:
        vfilter = 'pad=ceil(iw/2)*2:ceil(ih/2)*2'

    print(outname)
    if gpu_enabled:
        subprocess.run(['ffmpeg', '-y', '-hwaccel', 'auto', '-i', fname,
                        '-hide_banner', '-loglevel', 'error', # '-stats',
                        '-vcodec', 'h264_nvenc',
                        '-rc', 'constqp', '-qp', str(quality),
                        '-pix_fmt', 'yuv420p',
                        '-filter:v', vfilter,
                        outname])
    else:
        subprocess.run(['ffmpeg', '-y', '-i', fname,
                        '-hide_banner', '-loglevel', 'error', # '-stats',
                        '-vcodec', 'h264', '-qp', str(quality), '-pix_fmt', 'yuv420p',
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

    n_proc = config.get('encoding_nproc', 3)

    pool = Pool(n_proc)

    encoding_params = {
        'converted_video_speed': config['converted_video_speed'],
        'video_quality': config['video_quality'],
        'gpu_enabled': config['gpu_enabled']
    }

    for vidname in vidnames:
        basename = os.path.basename(vidname)
        base, ext = os.path.splitext(basename)
        outname = os.path.join(outpath, base+'.mp4')
        # if not (os.path.exists(outname) and same_length(vidname, outname)):
            # process_video(vidname, outname)
        pool.apply_async(process_video, (vidname, outname, encoding_params))

    pool.close()
    pool.join()

    return vidnames

def convert_all(config):

    process_all(config, process_folder)
