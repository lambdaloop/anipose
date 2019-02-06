# Anipose

Anipose is a framework for scalable [DeepLabCut](https://github.com/AlexEMG/DeepLabCut)-based analysis. It supports both 2d and 3d tracking, handles calibration and processing all files within a group of folders.

The name Anipose comes from **Ani**mal **Pose**, but it also sounds like "any pose".

(NOTE: This is not yet ready for production, still some issues to fix to make the experience smooth )

## Getting started

1) Setup DeepLabCut by following instruction [here](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md)
2) Install Anipose through pip: `pip install anipose`


## Input specification

Ideally someone who is collecting data should just be able to put all
the videos into a single folder during a data collection run.

Different data collection sessions (e.g. on different days) are usually
placed into different folders.

We'll need a config file to specify what to use for tracking and
post-processing. However, it's cumbersome to remember to put a new
config file for each new session, especially if the config doesn't
really change from session to session.

It would be nice to be able to place a separate config file into each
session though, to override the experiment config as needed.

It would be extra nice to reprocess the data if any of the config
parameters change.

layout of folder for videos for 2d tracking


```
.../experiment/config.toml
.../experiment/session1/videos-raw/vid1.avi
.../experiment/session1/videos-raw/vid2.avi
.../experiment/session2/videos-raw/vid1.avi
.../experiment/session2/videos-raw/vid2.avi
```

layout of folder for videos for 3d tracking

```
.../experiment/config.toml
.../experiment/session1/videos-raw/vid_abc_cam1.avi
.../experiment/session1/videos-raw/vid_abc_cam2.avi
.../experiment/session1/videos-raw/vid_abc_cam3.avi
.../experiment/session1/calibration/calib_v1_cam1.avi
.../experiment/session1/calibration/calib_v1_cam2.avi
.../experiment/session1/calibration/calib_v1_cam3.avi
```



## Configuration

Example config file for 2d/3d tracking:

```toml
# Project name
project = "flypose"

# Chnage this to match deeplabcut folder (one with trained network)
model_folder = '/Data/Videos/DLC_Analysis/Running-Brandon-2019-01-29'

# How many folders are nested in structure?
nesting = 1

# Settings for a threshold filter
# Removes data outside threshold (probably errors in tracking), and interpolates
filter_enabled = true
filter_medfilt = 13
filter_offset_threshold = 25
filter_score_threshold = 0.8
filter_spline = true

# labeling scheme...specify lines that you want to draw
[labeling]
scheme = [ ["head", "thorax", "abdomen"], ["thorax", "leg-1"] ]
```

## Output specification

The output structure should match the input structure, with additional
files resulting from the tracking.

The structure should be as follows:

/experiment/session/FOLDER

Where FOLDER is a folder storing the output of a specific processing
step.

It can be one of the following values:

  - **videos-raw** = input videos, compressed
  - **pose-2d** = 2d tracking for each of the input videos
  - **calibration** = camera parameters obtained from 3d calibration
  - **pose-3d** = 3d tracking for each group of input videos
  - **angles** = computed angles from 3d tracking
  - **videos-labeled** = videos labeled with the 2d tracking
  - **videos-3d** = 3d videos generated from 3d tracking
  - **config** = computed configuration for each session

## Outline of processing plan

For each experiment, for each session

1.  Compress the videos into videos-raw
2.  Place the configuration files into config (based on defaults and
    session config)
3.  Perform the 2d tracking based on the configuration
4.  Label the individual videos with 2d tracking
5.  If 3d tracking is enabled
    1.  Perform camera calibration
    2.  Perform triangulation of 2d tracking
    3.  Compute angles, if needed
    4.  Generate 3d videos

## Using the pipeline in the field

Ideally, there should be one repository with all the code, and the data
is held separate. Each data folder should come with a configuration file
of its own. The user should be able to invoke some pipeline script to
process everything, and separate pipeline scripts for each step.

Perhaps something like:

```
anipose calibrate # run calibration of intrinsics and extrinsics
anipose label # label the poses for each video
anipose label-videos # create videos for each pose
anipose run-data # run only the data portion (no viz)
anipose run-viz # run only the visualization pipeline
anipose run-all # run everything (run-data then run-viz)
```

The program anipose should parse out the config within the folder, and
figure out all the appropriate parameters to pass to the functions
underneath.

## Computing angle estimates

Although it’s very useful to get 3D tracking estimates, we also need
angle estimates.

However, not all angles make sense, it is up to the user to specify
which angles she cares about.

This may be specified in the config.toml file as follows:

```toml
[angles]
L1_CF = ["L1A", "L1B", "L1C"]
L1_FTi = ["L1B", "L1C", "L1D"]
L1_TiTa = ["L1C", "L1D", "L1E"]
```

The key above is the `[angles]` header, which specifies that whatever
follows is an angle.

Next, each angle is specified by a name on the left, and by a list of 3
joints on the right.

## Summarizing the data

After computing the whole pipeline for all videos, the final outputs of
interest (the 3d pose coordinates and angles, possibly the 2d
coordinates) are scattered across a lot of folders.

For further processing and analysis, it is often useful to have one
central file with all the data. Hence, Anipose provides the command
“summarize”, which summarizes everything into a “summaries” folder.
The output csv for each of angles, 3d, and 2d tracking coordinates has
all the data from all sessions, and a few extra columns to show where
the data comes from.

## Configuring the standardized 3D pose

In order to properly compare across different trials, different animals, and different setups, the 3D coordinates must be standardized relative to a common reference frame.

Anipose should allow configuration of this by specifying 2 sets of points to use as axes, and which axes these should be.

The algorithm to determine the axes is as follows:
- the first axis is taken as given
- the second axis is orthogonalized with respect to the first
- the third axis is the cross product of the first two axes

An axis is specified as a pair of points, with the axis going from the first to the second point.

Furthermore, it is often useful to set the zero to a standard reference point. Anipose allows this too.

An example configuration:
```toml
[triangulation]
axes = [
    ["x", "L1A", "L3A"],
    ["z", "L1B", "L1A"]
]
reference_point = "L1A"
```

## Calibration marker configuration

Anipose uses [ArUco markers](https://www.uco.es/investiga/grupos/ava/node/26) for
calibration. They are superior to checkerboards in that they are more
robust to blurring, rotation, and cropping. This makes them ideal for calibrating arbitrary camera setups.

In order to configure this, it should be possible to specify which
ArUco board was used to calibrate the cameras. What should be configurable:
- The type of board (ArUco / ChArUco)
- the size of the board (number squares in X and Y directions)
- ArUco marker dictionary (number of bits and number of markers in dictionary)
- Length of marker side in appropriate unit (triangulation is set to this unit)
- Length of marker separation (for ArUco) or square side (for ChArUco), in same unit


```toml
[calibration]
# aruco / charuco
board_type = "aruco"

# width and height of grid
board_size = [2, 2]

# number of bits in the markers
board_marker_bits = 5

# number of markers in dictionary (less is best)
board_marker_dict_number = 50

# length of marker side
board_marker_length = 4 # mm

# If aruco, length of marker separation
board_marker_separation_length = 1 # mm

# If charuco, square side length
# board_square_side_length = 8 # mm
```

## Calibration: how to actually calibrate
TODO: Document how to calibrate cameras

TODO: Document where to place the calibration folder and how this is processed

## Triangulation
TODO: document how to specify ROIs

TODO: provide example python/matlab code to automatically generate toml files

## References

Here are some references for DeepLabCut and other things this project relies upon:
```
@article{Mathisetal2018,
    title={DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
    author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
    journal={Nature Neuroscience},
    year={2018},
    url={https://www.nature.com/articles/s41593-018-0209-y}
}

@article{insafutdinov2016eccv,
    title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
    author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele},
    booktitle = {ECCV'16},
    url = {http://arxiv.org/abs/1605.03170}
}
```

