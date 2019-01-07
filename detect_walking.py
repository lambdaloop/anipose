#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy import signal, stats

def detect_walking(arr, fps, r_thres=0.5, pdiff_thres=0.6, min_count=5):
    """This function finds segments which correspond to walking, based on some simple heuristics.
Parameters
----------
arr : array representing the tracked x position of coxa-femur joint
fps : frames per second of recording

r_thres : threshold for template correlation
pdiff_thres : threshold for percent difference between start and end
min_count : min number of steps to include bout
 """
    nfilt = int(fps/30)
    arrf = signal.convolve(arr, np.ones(nfilt)/nfilt, mode='same')

    peaks, _ = signal.find_peaks(arrf, distance=fps/17)

    low,high = np.percentile(arrf, [30,60])
    target_range = high - low

    template = np.append(np.linspace(0,-1,40, endpoint=False),
                         np.linspace(-1,0,60))

    all_segments = []
    curr_start = None
    count = 0

    for p1, p2 in zip(peaks, peaks[1:]):
        test = signal.resample(arrf[p1:p2], len(template))
        r = stats.linregress(test, template).rvalue
        rang = np.max(test) - np.min(test)
        pdiff = np.abs(test[0] - test[-1])/rang

        # print(p1, p2, r, rang, pdiff)

        good = (r > r_thres) and (rang > target_range) \
            and (pdiff < pdiff_thres)

        if good and curr_start is None:
            curr_start = p1
            count = 1
        elif good:
            count += 1
        elif (not good) and curr_start is not None:
            if count >= min_count:
                segment = [curr_start, p1]
                all_segments.append(segment)
            curr_start = None
            count = 0

    if curr_start is not None and count >= min_count:
        segment = [curr_start, p1]
        all_segments.append(segment)

    return all_segments


if __name__ == '__main__':
    from glob import glob
    import os.path
    import matplotlib.pyplot as plt

    dirname = '../walking-tracking/06052018_fly2'
    FPS = 1000

    fnames = sorted(glob(os.path.join(dirname, '*.h5')))
    # videos = sorted(glob(os.path.join(dirname, '*.avi')))

    ## load the data
    fnum = 1
    fname = fnames[fnum]
    # vidname = videos[fnum]

    data = pd.read_hdf(fname)

    scorer = data.columns.levels[0][0]
    # bodyparts = [
    #     "body-coxa-left", "coxa-femur-left", "femur-tibia-left",
    #     "tibia-tarsus-left", "tarsus-end-left",
    #     "body-coxa-right", "coxa-femur-right", "femur-tibia-right",
    #     "tibia-tarsus-right", "tarsus-end-right",
    # ]

    liks = data[scorer].xs('likelihood', level='coords', axis=1)
    xs = data[scorer].xs('x', level='coords', axis=1)
    ys = data[scorer].xs('y', level='coords', axis=1)

    arr = np.array(xs['coxa-femur-left'])
    all_segments = detect_walking(arr, FPS)

    plt.figure(1)
    plt.clf()
    plt.plot(arr, alpha=0.2)
    for (start,end) in all_segments:
        plt.plot(np.arange(start,end),
                 arr[start:end])
    plt.ylim(np.percentile(arr, [0,100]))
    plt.draw()
    plt.show(block=False)
