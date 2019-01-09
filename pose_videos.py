#!/usr/bin/env python3

import os.path
import sys

## TODO: use deeplabcut v2 instead of this
pose_path = '/home/pierre/research/tuthill/DeepLabCut_pierre/pose-tensorflow'
# pose_path = '/home/tuthill/pierre/DeepLabCut_pierre/pose-tensorflow'

sys.path.append(pose_path)

# Deeper-cut dependencies
from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input

# Dependencies for video:
import pickle
import skimage.color
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import warnings
import cv2
from common import process_all

def getpose(image, net_cfg, outputs, outall=False):
    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, net_cfg)
    pose = predict.argmax_pose_predict(scmap, locref, net_cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose


def process_video(vidname, dataname, net_stuff):
    sess, inputs, outputs, net_cfg = net_stuff

    cap = cv2.VideoCapture(vidname)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    start = time.time()
    PredicteData = np.zeros((nframes, 3 * len(net_cfg['all_joints_names'])))

    # print("Starting to extract posture")
    for index in tqdm(range(nframes), ncols=70):
        ret, frame = cap.read()
        if not ret:
            break
        try:
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        except:
            break
        # image = img_as_ubyte(clip.get_frame(index * 1. / fps))
        pose = getpose(image, net_cfg, outputs)
        PredicteData[index, :] = pose.flatten()
        # NOTE: thereby net_cfg['all_joints_names'] should be same order as bodyparts!

    cap.release()
    stop = time.time()

    dictionary = {
        "start": start,
        "stop": stop,
        "run_duration": stop - start,
        "Scorer": scorer,
        "config file": net_cfg,
        "fps": fps,
        "frame_dimensions": (height, width),
        "nframes": nframes
    }
    metadata = {'data': dictionary}

    pdindex = pd.MultiIndex.from_product(
        [net_cfg['all_joints_names'], ['x', 'y', 'likelihood']],
        names=['bodyparts', 'coords'])

    # print("Saving results...")
    DataMachine = pd.DataFrame(
        PredicteData, columns=pdindex, index=range(nframes))
    DataMachine.to_hdf(
        dataname, 'df_with_missing', format='table', mode='w')
    with open(os.path.splitext(dataname)[0] + '_metadata.pickle',
              'wb') as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


def process_session(config, session_path, net_stuff):
    pipeline_videos_raw = config['pipeline_videos_raw']
    pipeline_pose = config['pipeline_pose_2d']

    videos = glob(os.path.join(session_path, pipeline_videos_raw, '*.avi'))
    videos = sorted(videos)

    for video in videos:
        basename = os.path.basename(video)
        basename, _ = os.path.splitext(basename)
        os.makedirs(os.path.join(session_path, pipeline_pose), exist_ok=True)
        dataname_base = basename + '.h5'
        dataname = os.path.join(session_path, pipeline_pose, dataname_base)
        print(dataname)
        try:
            # Attempt to load data...
            pd.read_hdf(dataname)
            # print("Video already analyzed!", dataname)
        except:
            # print("Loading ", video)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # print('reprocess', video)
                process_video(video, dataname, net_stuff)



def pose_videos_all(config):
    pipeline_prefix = config['path']

    model_path = os.path.join(config['model_folder'], config['model_name'])

    net_cfg = load_config(os.path.join(model_path, 'test', "pose_cfg.yaml"))

    net_cfg['init_weights'] = os.path.join(model_path, 'train',
                                           'snapshot-{}'.format(
                                               config['model_train_iter']))

    scorer = 'DeepCut_{}_{}'.format(config['model_name'], config['model_train_iter'])
    sess, inputs, outputs = predict.setup_pose_prediction(net_cfg)

    net_stuff = sess, inputs, outputs, net_cfg

    process_all(config, process_session, net_stuff)
