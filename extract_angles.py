#!/usr/bin/env python3

import sys, os

## add parent directory
subfolder = os.getcwd().split('pipeline')[0]
sys.path.append(subfolder)

from myconfig_pipeline import pipeline_videos_raw, pipeline_pose, pipeline_angles

import numpy as np
import pandas as pd
from glob import glob
import os.path
# import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import splev, splrep
import sys
import re
import cv2
from detect_walking import detect_walking


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def get_bouts_tracking(fname, meta):
    data = pd.read_hdf(fname)

    outname = os.path.basename(fname.split('DeepCut')[0])
    match = re.search(r'_fly_(\d+)', outname)
    if match:
        flynum = int(match.groups()[0])
    else:
        flynum = 1

    # meta = metadict[outname+'.avi']
    fps = meta['fps']

    scorer = data.columns.levels[0][0]
    bodyparts = [
        "body-coxa-left", "coxa-femur-left", "femur-tibia-left",
        "tibia-tarsus-left", "tarsus-end-left",
        "body-coxa-right", "coxa-femur-right", "femur-tibia-right",
        "tibia-tarsus-right", "tarsus-end-right",
    ]

    liks = data[scorer].xs('likelihood', level='coords', axis=1)
    xs = data[scorer].xs('x', level='coords', axis=1)
    ys = data[scorer].xs('y', level='coords', axis=1)

    arr = np.array(xs['coxa-femur-left'])
    all_segments = detect_walking(arr, fps)

    all_data = []

    for start, end in all_segments:
        good = np.arange(start, end)

        ## load the data
        # out_fname = '{}_frames-{:05d}-{:05d}'.format(outname, start,end)
        # out_fname = os.path.join(outdir, out_fname)
        # print(out_fname)
        print(start, end)

        data = pd.read_hdf(fname)

        scorer = data.columns.levels[0][0]
        bodyparts = [
            "body-coxa-left", "coxa-femur-left", "femur-tibia-left",
            "tibia-tarsus-left", "tarsus-end-left",
            "body-coxa-right", "coxa-femur-right", "femur-tibia-right",
            "tibia-tarsus-right", "tarsus-end-right",
        ]

        liks = data[scorer].xs('likelihood', level='coords', axis=1)
        xs = data[scorer].xs('x', level='coords', axis=1)
        ys = data[scorer].xs('y', level='coords', axis=1)

        times = np.arange(0, len(data)) / fps

        ## exclude possibly bad labels
        xs[liks < 0.8] = np.nan
        ys[liks < 0.8] = np.nan

        # nsmooth = int(round(fps/70))
        # nsmooth = nsmooth + (nsmooth % 2)-1
        # nsmooth = max(nsmooth, 3)
        nsmooth = 11

        ## exclude jumps from one leg to another
        for _ in range(2):
            for bp in bodyparts:
                xvals = np.array(xs[bp])
                yvals = np.array(ys[bp])

                nans, x = nan_helper(xvals)
                mx = np.zeros(xvals.shape) * np.nan
                mx[~nans] = signal.medfilt(xvals[~nans],kernel_size=nsmooth)
                dx = np.abs(xvals - mx)

                nans, x = nan_helper(yvals)
                my = np.zeros(yvals.shape) * np.nan
                my[~nans] = signal.medfilt(yvals[~nans],kernel_size=nsmooth)
                dy = np.abs(yvals - my)

                with np.errstate(invalid='ignore'):
                    bad = (dx > (fps/100)) | (dy > (fps/100))
                    
                xs.loc[bad, bp] = np.nan
                ys.loc[bad, bp] = np.nan


        xsf = xs.copy()
        ysf = ys.copy()

        bad_percent = np.mean(np.mean(np.isnan(xsf)))
        if bad_percent > 0.8: # too much bad data
            print('^ too much bad data, ignoring')
            continue
        
        for bp in bodyparts:
            # print(bp)
            xvals = np.array(xsf[bp])
            nans, x= nan_helper(xvals)
            if np.sum(nans) > 1:
                # xvals[nans]= splev(x(nans), splrep(x(~nans), xvals[~nans], k=3, s=3))
                # print(bp, nans[:10])
                xvals[nans]= np.interp(x(nans), x(~nans), xvals[~nans])
            xsf[bp] = xvals

            yvals = np.array(ysf[bp])
            nans, x= nan_helper(yvals)
            if np.sum(nans) > 1:
                # yvals[nans]= splev(x(nans), splrep(x(~nans), yvals[~nans], k=3, s=3))
                yvals[nans]= np.interp(x(nans), x(~nans), yvals[~nans])
            ysf[bp] = yvals

        angles = dict()
        for leg in ['left', 'right']:

            ## extract angle
            v0 = np.array([xsf['coxa-femur-'+leg]-xsf['body-coxa-'+leg],
                           ysf['coxa-femur-'+leg]-ysf['body-coxa-'+leg]]).T
            v1 = np.array([xsf['femur-tibia-'+leg]-xsf['coxa-femur-'+leg],
                           ysf['femur-tibia-'+leg]-ysf['coxa-femur-'+leg]]).T
            v2 = np.array([xsf['tibia-tarsus-'+leg]-xsf['femur-tibia-'+leg],
                           ysf['tibia-tarsus-'+leg]-ysf['femur-tibia-'+leg]]).T
            v3 = np.array([xsf['tarsus-end-'+leg]-xsf['tibia-tarsus-'+leg],
                           ysf['tarsus-end-'+leg]-ysf['tibia-tarsus-'+leg]]).T


            angle_body_coxa = np.arctan2(v0[:, 1], v0[:, 0])
            angle_coxa_femur = np.arctan2(v1[:, 1], v1[:, 0]) - np.arctan2(-v0[:, 1], -v0[:, 0])
            angle_femur_tibia = np.arctan2(v2[:, 1], v2[:, 0]) - np.arctan2(-v1[:, 1], -v1[:, 0])
            angle_tibia_tarsus = np.arctan2(v3[:, 1], v3[:, 0]) -np.arctan2(-v2[:, 1], -v2[:, 0])

            # angle_tibia_tarsus[angle_tibia_tarsus < np.pi/2] = np.nan

            angle_coxa_femur = np.mod(np.pi*2 - angle_coxa_femur, np.pi*2)
            angle_tibia_tarsus = np.mod(np.pi*2 - angle_tibia_tarsus, np.pi*2)

            ## exclude bad angles
            with np.errstate(invalid='ignore'):
                angle_coxa_femur[angle_coxa_femur > np.pi] = np.nan
                angle_femur_tibia[angle_femur_tibia < 0] = np.nan
                angle_femur_tibia[angle_femur_tibia > np.pi] = np.nan
                angle_tibia_tarsus[np.isnan(angle_femur_tibia)] = np.nan

            ## exclude bad data
            xsf.loc[np.isnan(angle_coxa_femur), 'femur-tibia-'+leg] = np.nan
            xsf.loc[np.isnan(angle_coxa_femur), 'coxa-femur-'+leg] = np.nan
            xsf.loc[np.isnan(angle_femur_tibia), 'coxa-femur-'+leg] = np.nan
            xsf.loc[np.isnan(angle_femur_tibia), 'femur-tibia-'+leg] = np.nan
            xsf.loc[np.isnan(angle_femur_tibia), 'tibia-tarsus-'+leg] = np.nan
            xsf.loc[np.isnan(angle_tibia_tarsus), 'tarsus-end-'+leg] = np.nan

            ysf[np.isnan(xsf)] = np.nan
            xs[np.isnan(xsf)] = np.nan
            ys[np.isnan(ysf)] = np.nan

            ## setup angles
            angles_new = {
                'angle-body-coxa-'+leg: angle_body_coxa,
                'angle-coxa-femur-'+leg: angle_coxa_femur,
                'angle-femur-tibia-'+leg: angle_femur_tibia,
                'angle-tibia-tarsus-'+leg: angle_tibia_tarsus
            }

            angles = dict(angles, **angles_new)


        ## testing plot
        # plt.clf()
        # plt.plot(angle_body_coxa)
        # plt.plot(angle_coxa_femur)
        # plt.plot(angle_femur_tibia)
        # plt.plot(angle_tibia_tarsus)
        # plt.legend(labels=['body-coxa', 'coxa-femur', 'femur-tibia', 'tibia-tarsus'])
        # plt.draw()
        # plt.show(block=False)



        ## interpolate bad data using cubic splines
        anglesf = dict()

        for key, ang in angles.items():
            # print(key)
            angf = np.copy(ang)
            nans, x= nan_helper(ang)
            # anglef[nans]= np.interp(x(nans), x(~nans), angle[~nans])
            if np.sum(nans) > 0:
                angf[nans]= splev(x(nans), splrep(x(~nans), ang[~nans], k=3, s=3))
            anglesf[key+'-intp'] = angf


        ## write data to csv
        dx = pd.DataFrame(xs.copy())
        dx.columns = [c + '-x' for c in xs.columns]

        dy = pd.DataFrame(ys.copy())
        dy.columns = [c + '-y' for c in ys.columns]

        dxf = pd.DataFrame(xsf.copy())
        dxf.columns = [c + '-x-intp' for c in xs.columns]

        dyf = pd.DataFrame(ysf.copy())
        dyf.columns = [c + '-y-intp' for c in ys.columns]

        d_angle = pd.concat([pd.DataFrame(angles), pd.DataFrame(anglesf)],
                            axis=1) * 180 / np.pi

        dw = pd.concat([dx, dy, dxf, dyf, d_angle], axis=1)
        dw['frame'] = dw.index
        dw['id'] = outname
        dw['start'] = start
        dw['end'] = end
        dw['id_clip'] = '{}_frames-{:05d}-{:05d}'.format(outname, start, end)
        dw['flynum'] = flynum
        for key, value in meta.items():
            dw[key] = value

        dw = dw.iloc[good]

        all_data.append(dw)

    if len(all_data) > 0:
        all_df = pd.concat(all_data)
        return all_df
    else:
        return None

videofolders = next(os.walk(pipeline_pose))[1]
videofolders = sorted(videofolders)

for videofolder in videofolders:
    outdir = os.path.join(pipeline_angles, videofolder)
    os.makedirs(outdir, exist_ok=True)

    fnames = sorted(glob(os.path.join(pipeline_pose, videofolder, '*.h5')))

    for fname in fnames:
        basename = os.path.basename(fname.split('_DeepCut')[0])
        out_fname = os.path.join(pipeline_angles, videofolder, basename+'.csv')

        if os.path.exists(out_fname):
            continue
        
        vidname = os.path.join(pipeline_videos_raw, videofolder, basename+'.avi')

        print(vidname)

        meta = dict()
        meta['session'] = videofolder

        cap = cv2.VideoCapture(vidname)
        meta["framecount"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta["fps"] = cap.get(cv2.CAP_PROP_FPS)
        cap.release()


        df = get_bouts_tracking(fname, meta)
        if df is None:
            df = pd.DataFrame([meta])

        df.to_csv(out_fname, index=False)
