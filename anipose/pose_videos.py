#!/usr/bin/env python3

import os.path

# Dependencies for video:
import os
from glob import glob
import io
from contextlib import redirect_stdout
import shutil

from .common import natural_keys, make_process_fun

def rename_dlc_files(folder, base):
    files = glob(os.path.join(folder, base+'*'))
    for fname in files:
        basename = os.path.basename(fname)
        _, ext = os.path.splitext(basename)
        os.rename(os.path.join(folder, basename),
                  os.path.join(folder, base + ext))


def move_sleap_predictions(video, outdir):
    basename = os.path.basename(video)
    basename, ext = os.path.splitext(basename)
    source_dir = os.path.dirname(video)
    pred_name = basename + '.predictions.slp'
    source_path = os.path.join(source_dir, pred_name)
    dest_path = os.path.join(outdir, pred_name)
    shutil.move(source_path, dest_path)
    

def process_session(config, session_path):
    pipeline_videos_raw = config['pipeline']['videos_raw']
    pipeline_pose = config['pipeline']['pose_2d']
    video_ext = config['video_extension']
    model_type = config['model_type']


    source_folder = os.path.join(session_path, pipeline_videos_raw)
    outdir = os.path.join(session_path, pipeline_pose)

    videos = glob(os.path.join(source_folder, '*.'+video_ext))
    videos = sorted(videos, key=natural_keys)

    if len(videos) > 0:
        os.makedirs(outdir, exist_ok=True)

    videos_to_process = []
    for video in videos:
        basename = os.path.basename(video)
        basename, ext = os.path.splitext(basename)

        if model_type == 'deeplabcut':
            dataname = os.path.join(outdir, basename + '.h5')
        elif model_type == 'sleap':
            dataname = os.path.join(outdir, basename + '.predictions.slp')
            
        if os.path.exists(dataname):
            continue
        else:
            if model_type == 'deeplabcut':
                rename_dlc_files(outdir, basename) # try renaming in case it processed before
            if os.path.exists(dataname):
                print(video)
                continue
            else:
                videos_to_process.append(video)

    if len(videos_to_process) > 0:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if model_type == 'deeplabcut':
            import deeplabcut
            config_name = os.path.join(config['model_folder'], 'config.yaml')
            trap = io.StringIO()
            for i in range(0, len(videos_to_process), 5):
                batch = videos_to_process[i:i+5]
                for video in batch:
                    print(video)
                with redirect_stdout(trap):
                    deeplabcut.analyze_videos(config_name, batch,
                                              videotype=video_ext, save_as_csv=False,
                                              destfolder=outdir, TFGPUinference=False)
                for video in batch:
                    basename = os.path.basename(video)
                    basename, ext = os.path.splitext(basename)
                    rename_dlc_files(outdir, basename)
        elif model_type == 'sleap':
            from sleap_nn.predict import run_inference
            model_paths = glob(os.path.join(config['model_folder'], '*'))
            for video in videos_to_process:
                print(video)
                labels = run_inference(
                    data_path=video,
                    model_paths=model_paths,
                    return_confmaps=False,
                    peak_threshold=0.1
                )
                move_sleap_predictions(video, outdir)

pose_videos_all = make_process_fun(process_session)

