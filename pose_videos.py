"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu
P Karashchuk, pierrek@uw.edu

This script analyzes videos based on a trained network.
You need tensorflow for evaluation. Run by:
CUDA_VISIBLE_DEVICES=0 python3 AnalyzeVideos.py

"""

####################################################
# Dependencies
####################################################

import os.path
import sys

subfolder = os.getcwd().split('pipeline')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "pose-tensorflow/")
sys.path.append(subfolder + "Generating_a_Training_Set")

from myconfig_pipeline import cropping, Task, date, \
    trainingsFraction, resnet, trainingsiterations, snapshotindex, shuffle,x1, x2, y1, y2
from myconfig_pipeline import pipeline_prefix, pipeline_videos_raw, pipeline_pose

# Deep-cut dependencies
from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input

# Dependencies for video:
import pickle
# import matplotlib.pyplot as plt
import imageio
imageio.plugins.ffmpeg.download()
import skimage.color
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import warnings
import cv2

def getpose(image, cfg, outputs, outall=False):
    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose


####################################################
# Loading data, and defining model folder
####################################################

basefolder = '../pose-tensorflow/models/'  # for cfg file & ckpt!
modelfolder = (basefolder + Task + str(date) + '-trainset' +
               str(int(trainingsFraction * 100)) + 'shuffle' + str(shuffle))
cfg = load_config(modelfolder + '/test/' + "pose_cfg.yaml")

##################################################
# Load and setup CNN part detector
##################################################

# Check which snap shots are available and sort them by # iterations
Snapshots = np.array([
    fn.split('.')[0]
    for fn in os.listdir(modelfolder + '/train/')
    if "index" in fn
])
increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]

print(modelfolder)
print(Snapshots)

##################################################
# Compute predictions over images
##################################################

# Check if data already was generated:
cfg['init_weights'] = modelfolder + '/train/' + Snapshots[snapshotindex]

# Name for scorer:
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(
    date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)
cfg['init_weights'] = modelfolder + '/train/' + Snapshots[snapshotindex]

sess, inputs, outputs = predict.setup_pose_prediction(cfg)

pdindex = pd.MultiIndex.from_product(
    [cfg['all_joints_names'], ['x', 'y', 'likelihood']],
    names=['bodyparts', 'coords'])

##################################################
# Datafolder
##################################################

def process_video(vidname, dataname):
    # clip = VideoFileClip(vidname)
    cap = cv2.VideoCapture(vidname)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start = time.time()
    PredicteData = np.zeros((nframes, 3 * len(cfg['all_joints_names'])))

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
        pose = getpose(image, cfg, outputs)
        PredicteData[index, :] = pose.flatten()
        # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!

    cap.release()
    stop = time.time()

    dictionary = {
        "start": start,
        "stop": stop,
        "run_duration": stop - start,
        "Scorer": scorer,
        "config file": cfg,
        "fps": fps,
        "frame_dimensions": (height, width),
        "nframes": nframes
    }
    metadata = {'data': dictionary}

    # print("Saving results...")
    DataMachine = pd.DataFrame(
        PredicteData, columns=pdindex, index=range(nframes))
    DataMachine.to_hdf(
        dataname, 'df_with_missing', format='table', mode='w')
    with open(os.path.splitext(dataname)[0] + '_metadata.pickle',
              'wb') as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

def get_folders(path):
    folders = next(os.walk(path))[1]
    return sorted(folders)

        
experiments = get_folders(pipeline_prefix)

for exp in experiments:
    exp_path = os.path.join(pipeline_prefix, exp)
    sessions = get_folders(exp_path)
    
    for session in sessions:
        print(session)

        videos = glob(os.path.join(pipeline_prefix, exp, session, pipeline_videos_raw, 'vid'+'*.avi'))
        videos = sorted(videos)

        for video in videos:
            basename = os.path.basename(video)
            basename, _ = os.path.splitext(basename)
            os.makedirs(os.path.join(pipeline_prefix, exp, session, pipeline_pose), exist_ok=True)
            dataname_base = basename + '.h5'
            dataname = os.path.join(pipeline_prefix, exp, session, pipeline_pose, dataname_base)
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
                    process_video(video, dataname)

