#!/usr/bin/env python3

import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy import signal, stats
import numpy as np
from pyzbar import pyzbar
from collections import defaultdict, Counter

import numpy as np
import pickle


cam_intrinsics = {
    "C": (np.array([[21890.40164506845, 0.0, 169.40772844173478],
                    [0.0, 25966.39972124997, 450.3353829454416],
                    [0.0, 0.0, 1.0]]),
          np.array([-25.287440661876197, 6414.822594324628, 0.06830900427998216,
                    0.15829293043889087, 1.8243621200236957]) ),
    "B": (np.array([[33877.80908698013, 0.0, 644.412517857351],
                    [0.0, 36288.31284658607, 527.4133303148893],
                    [0.0, 0.0, 1.0]]),
          np.array([25.311397100210282, -2.0450400994346007, 0.09073000146198174,
                    0.1127783451139442, -0.00101267594575803]) ),
    "A": (np.array([[52843.813795162634, 0.0, 639.7051408205888],
                    [0.0, 67384.96597208804, 523.6823574850912],
                    [0.0, 0.0, 1.0]]),
          np.array([122.63056222554557, -0.3050421546786041, -0.12494052984376995,
                    -0.01128374233060388, -5.6104480867203854e-05]) )
}


## not sure butt one got good values
# cam_intrinsics["butt"] = cam_intrinsics["front"]

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.GridBoard_create(2, 2, 4, 1, dictionary)

prefix = "/home/pierre/research/tuthill/walking-videos-compressed"
# fname = "test/2018-09-11/2018-09-11--10-49-34_{}.avi"
# fname = "test/2018-09-11/2018-09-11--10-52-57_{}.avi"
# fname = "test2/2018-09-18/calib_2018-09-18--15-31-01_{}.avi"
fname = "test2/2018-09-21/calib_2018-09-21--13-44-28_{}.avi"

# prefix = "/home/pierre/research/tuthill/videos-others-compressed/videos-evyn-compressed/3d-tracking-calibration"
# fname = "18-Sep-2018 11 {}.avi"


def detect_aruco(gray, intrinsics):
    # grayb = gray
    grayb = cv2.GaussianBlur(gray, (5, 5), 0)

    params = aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 100
    params.adaptiveThreshWinSizeMax = 600
    params.adaptiveThreshWinSizeStep = 50
    params.adaptiveThreshConstant = 5

    corners, ids, rejectedImgPoints = aruco.detectMarkers(grayb, dictionary,  parameters=params)

    INTRINSICS_K, INTRINSICS_D = intrinsics
    
    if ids is None:
        return [], []
    elif len(ids) < 2:
        return corners, ids
    
    detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
        aruco.refineDetectedMarkers(grayb, board, corners, ids,
                                    rejectedImgPoints,
                                    INTRINSICS_K, INTRINSICS_D,
                                    parameters=params)

    return detectedCorners, detectedIds

def estimate_pose(gray, intrinsics):

    detectedCorners, detectedIds = detect_aruco(gray, intrinsics)
    if len(detectedIds) < 3:
        return False, None

    INTRINSICS_K, INTRINSICS_D = intrinsics

    ret, rvec, tvec = aruco.estimatePoseBoard(detectedCorners, detectedIds, board,
                                              INTRINSICS_K, INTRINSICS_D)

    # flip the orientation as needed
    a,b,c = rvec[:, 0]
    # if np.sign(np.product(rvec)) < 0 and (b < 0 or c < 0):
    if np.sign(np.product(rvec)) < 0 and a < 0:
        rvec[1,0] = -rvec[1,0]
        rvec[0,0] = -rvec[0,0]

    return True, (detectedCorners, detectedIds, rvec, tvec)


def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(rvec)
    out[:3,:3] = rotmat
    out[:3, 3] = tvec.flatten()
    out[3, 3] = 1
    return out


def mean_transform(M_list):
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in M_list]
    tvecs = [M[:3, 3] for M in M_list]

    rvec = np.median(rvecs, axis=0)
    tvec = np.median(tvecs, axis=0)

    return make_M(rvec, tvec)

cam_names = ["A", "B", "C"]
cam_mapping = dict([(x,x) for x in cam_names])

# cam_names = ["front", "butt", "side"]
# cam_mapping = {"front": "A", "butt": "B", "side": "C"}


cam_name = "A"

cap = cv2.VideoCapture(os.path.join(prefix, fname.format(cam_name)))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start = 0
cap.set(1, start)

go = 20

all_vecs = []

# for framenum in trange(length-start):
framenum = 1000

cap.set(1,framenum)
ret, frame = cap.read()
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)        

# if framenum % 20 != 0 and go <= 0:
#     continue

intrinsics = cam_intrinsics[cam_name]

success, result = estimate_pose(gray, intrinsics)
print(success)
# if not success:
#     continue
# else:
#     go = 20

corners, ids, rvec, tvec = result

# all_vecs.append(result)

go = max(0, go-1)

print(rvec)

plt.figure(1)
plt.clf()
plt.imshow(frame)
plt.draw()
plt.show(block=False)


