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
from scipy.ndimage.filters import gaussian_filter
from scipy.special import logsumexp

lk_params = dict( winSize  = (18,18),
                  maxLevel = 6,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

farneback_params = dict(
    pyr_scale=0.5, levels=5,
    winsize=7, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
)

stride = 8

point_sd = 6
move_sd = 5

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


def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    out = (Ia*wa) + (Ib*wb) + (Ic*wc) + (Id*wd)

    return out

def find_good_locs(locs, probs, width, height):
    good = probs > 1e-20
    good &= locs[:, 0] > 1
    good &= locs[:, 0] < width-1
    good &= locs[:, 1] > 1
    good &= locs[:, 1] < height-1
    return np.where(good)[0]


def get_grid_probs_locs(locs, probs, imgsize, blur=True, blur_sd=point_sd):
    ## alternative code, more precise but slower
    # out = np.zeros(grid.shape[0])
    # for i in range(locs.shape[0]):
    #     if probs[i] > 0.01:
    #         check_x, check_y = np.int32(np.round(locs[i]/step)*step)
    #         good = good_x[check_x] & good_y[check_y]
    #         dists = np.linalg.norm(locs[i] - grid[good], axis=1)
    #         check = dists < point_sd*2.5
    #         check_full = np.arange(len(grid))
    #         check_full = check_full[good][check]
    #         out[check_full] += stats.norm.pdf(dists[check], scale=point_sd) * probs[i]

    height, width = imgsize
    
    out = np.zeros(height*width)
    good = find_good_locs(locs, probs, width, height)
    rounded = np.int32(np.round(locs[good]))
    x, y = rounded.T
    index = y*width + x
    for ix, px in zip(index, good):
        out[ix] += probs[px]
    prob_img = out.reshape(height, width)

    if blur:
        prob_img = gaussian_filter(prob_img, sigma=blur_sd, truncate=3.5)
        # n = int(round(blur_sd * 4))
        # if n % 2 == 0: n += 1
        # prob_img = cv2.GaussianBlur(prob_img, (n, n), blur_sd)

    prob_img += 1e-100
    prob_img /= np.sum(prob_img)
    
    return prob_img
    


def get_grid_probs_frame(scmap, locref, imgsize, partnum):
    probs = scmap[:, :, partnum]
    offmat = locref[:, :, partnum]
    
    pflat = probs.reshape(-1)
    pflat = pflat / np.sum(pflat)

    ys, xs = np.unravel_index(range(probs.size), probs.shape)
    xs_img = xs * stride + stride*0.5
    ys_img = ys * stride + stride*0.5
    pos_start = np.array([xs_img, ys_img]).T

    pos_end = pos_start + offmat.reshape(-1, 2)

    prob_img = get_grid_probs_locs(pos_end, pflat, imgsize)
    
    return prob_img

def get_flow(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                        None, **farneback_params)
    return flow


def move_probs_flow(prob_img, flow):
    h, w = prob_img.shape
    ys, xs = np.mgrid[0:h, 0:w]

    prob_img_new = bilinear_interpolate_numpy(prob_img, xs-flow[:,:,0], ys-flow[:, :, 1])
    prob_img_new[prob_img_new < 0] = 0
    prob_img_new = gaussian_filter(prob_img_new, sigma=2, truncate=3.5)

    return prob_img_new / np.sum(prob_img_new)

    
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

    nframes = min(nframes, 1000)
    
    start = time.time()
    nparts = len(cfg['all_joints_names'])
    PredicteData = np.zeros((nframes, 3 * nparts))
    PredicteDataFlow = np.zeros((nframes, 3 * nparts))

    prev_gray = None
    prob_img = None
    prob_img_flow = np.zeros((nparts, height, width))
    weights = np.zeros((nparts, height, width))
    pose_flow = np.zeros((nparts, 3))
    
    # print("Starting to extract posture")
    for index in tqdm(range(nframes), ncols=70):
        ret, frame = cap.read()
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break

        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        scmap, locref, pose = getpose(image, cfg, outputs, outall=True)
        PredicteData[index, :] = pose.flatten()

        imgsize = curr_gray.shape
        for ix_bp in range(nparts):
            w = get_grid_probs_frame(scmap, locref, imgsize, ix_bp)
            wlog = np.log(w + 0.01)
            weights[ix_bp] = wlog - logsumexp(wlog)


        if prob_img is not None:
            flow = get_flow(prev_gray, curr_gray)
            for ix_bp in range(nparts):
                prob_img_flow[ix_bp] = move_probs_flow(np.exp(prob_img[ix_bp]), flow)
            prob_img_flow = np.log(prob_img_flow+1e-100)
            prob_img_next = prob_img_flow + weights
        else:
            prob_img_next = weights

        prob_img_next[prob_img_next < -60] = -60
        for ix_bp in range(nparts):
            prob_img_next[ix_bp] = prob_img_next[ix_bp] - logsumexp(prob_img_next[ix_bp])

            y, x = np.unravel_index(np.argmax(prob_img_next[ix_bp]), prob_img_next[ix_bp].shape)
            pmax = np.exp(prob_img_next[ix_bp, y, x])
            
            pose_flow[ix_bp] = [x, y, pmax]

        PredicteDataFlow[index, :] = pose_flow.flatten()

        prev_gray = curr_gray
        prob_img = prob_img_next

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
        dataname + '.h5', 'df_with_missing', format='table', mode='w')

    DataMachineFlow = pd.DataFrame(
        PredicteDataFlow, columns=pdindex, index=range(nframes))
    DataMachineFlow.to_hdf(
        dataname + '_flow.h5', 'df_with_missing', format='table', mode='w')

    with open(dataname + '_metadata.pickle',
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
            dataname_base = basename
            dataname = os.path.join(pipeline_prefix, exp, session, pipeline_pose, dataname_base)
            print(dataname)
            try:
                # Attempt to load data...
                pd.read_hdf(dataname + '.h5')
                # print("Video already analyzed!", dataname)
            except:
                # print("Loading ", video)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # print('reprocess', video)
                    process_video(video, dataname)

