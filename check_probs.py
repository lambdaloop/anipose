"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu
P Karashchuk, pierrek@uw.edu

This script outputs probability maps for a video
You need tensorflow for evaluation. Run by:
CUDA_VISIBLE_DEVICES=0 python3 check_probs.py vidname outfolder log

"""

####################################################
# Dependencies
####################################################

import os.path
import sys

subfolder = os.getcwd().split('Evaluation-Tools')[0]
# subfolder = os.getcwd().split('pipeline')[0]
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
sys.path.append(subfolder + "pose-tensorflow/")
sys.path.append(subfolder + "Generating_a_Training_Set")

from myconfig_analysis import Task, date, \
    trainingsFraction, resnet, trainingsiterations, snapshotindex, shuffle
# from myconfig_pipeline import pipeline_prefix, pipeline_videos_raw, pipeline_pose

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




bodyparts = cfg['all_joints_names']

def cmap_image(img, cmap):
    vnorm = (img - np.min(img)) / (np.max(img) - np.min(img))
    out = cmap(vnorm, bytes=True)[:,:,:3]
    return out


def process_video(vidname, outfolder, transform='log'):
    
    cap = cv2.VideoCapture(vidname)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cmap = get_cmap('inferno')
    nparts = len(bodyparts)

    basename = os.path.basename(vidname)
    basename = os.path.splitext(basename)[0]

    
    for partnum in range(nparts):
        print('part {}/{} -- {}'.format(partnum+1, nparts, bodyparts[partnum]))
        
        cap.set(1,0)
        
        outname = os.path.join(outfolder, '{}-{}-{}.avi'.format(
            basename, transform, bodyparts[partnum]))
        
        writer = skvideo.io.FFmpegWriter(outname, inputdict={
            '-hwaccel': 'auto',
            '-framerate': str(fps),
        }, outputdict={
            '-vcodec': 'h264_nvenc', '-qp': '30'
        })
    
        for index in tqdm(range(nframes), ncols=70):

            ret, frame = cap.read()
            if not ret:
                break
            try:
                image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            except:
                break


            image_batch = data_to_input(image)
            outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
            scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
            pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

            dsize = image.shape[:2][::-1]
            low_cutoff = 1e-10

            vals = scmap[:, :, partnum]
            
            vals[vals < low_cutoff] = low_cutoff
            if transform == 'log':
                vals = np.log(vals)
            elif transform == 'identity':
                vals = vals
                
            vals = cv2.resize(vals,
                              dsize=dsize,
                              interpolation=cv2.INTER_NEAREST)
            imtest = cmap_image(vals, cmap)

            imout = np.uint8(imtest*0.5 + image*0.5)
            pos = tuple(np.int32(pose[partnum,:2]))
            imout = cv2.circle(imout, pos, 3, (255,255,255), -1)
            imout = cv2.circle(imout, pos, 5, (0,0,0), 2)


            writer.writeFrame(imout)

        writer.close()
    cap.release()


def get_folders(path):
    folders = next(os.walk(path))[1]
    return sorted(folders)


if __name__ == '__main__':
    process_video(sys.argv[1], sys.argv[2], sys.argv[3])
