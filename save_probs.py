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
import skvideo.io
from matplotlib.pyplot import get_cmap

def getpose(image, cfg, outputs, outall=False):
    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(image)
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


vidname = '/home/pierre/research/tuthill/flywalk-pipeline-new/test2/2018-09-21/videos-raw/vid_2018-09-21--13-39-24_A.avi'

bodyparts = cfg['all_joints_names']

def cmap_image(img, cmap):
    vnorm = (img - np.min(img)) / (np.max(img) - np.min(img))
    out = cmap(vnorm, bytes=True)[:,:,:3]
    return out


def process_video(vidname, outfolder):
    
    cap = cv2.VideoCapture(vidname)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cmap = get_cmap('inferno')
    nparts = len(bodyparts)

    basename = os.path.basename(vidname)
    basename = os.path.splitext(basename)[0]

    cap.set(1,0)

    outname = os.path.join(outfolder, '{}.npz'.format(basename))

    ret, frame = cap.read()

    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    image_batch = data_to_input(image)
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

    scmap_all = np.zeros((nframes,)+scmap.shape)
    locref_all = np.zeros((nframes,)+locref.shape)
    pose_all = np.zeros((nframes,)+pose.shape)
    

    for index in tqdm(range(nframes), ncols=70):

        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        image_batch = data_to_input(image)
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

        scmap_all[index] = scmap
        locref_all[index] = locref
        pose_all[index] = pose
        

    cap.release()

    np.savez_compressed(outname, scmap=scmap_all, locref=locref_all, pose=pose_all)
    


if __name__ == '__main__':
    process_video(sys.argv[1], sys.argv[2])
