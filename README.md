# Anipose

Anipose is a framework for scalable DeepLabCut-based analysis. It supports both 2d and 3d tracking, handles calibration and processing all files within a group of folders. 

The name Anipose comes from *Ani*mal *Pose*, but it also sounds like "any pose".

(NOTE: This is not yet ready for production, still some issues to fix to make the experience smooth )

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
/experiment/config.toml
/experiment/session1/vid1.avi
/experiment/session1/vid2.avi
/experiment/session2/vid1.avi
/experiment/session2/vid2.avi
```

layout of folder for videos for 3d tracking

```
/experiment/config.toml
/experiment/session1/vid_abc_cam1.avi
/experiment/session1/vid_abc_cam2.avi
/experiment/session1/vid_abc_cam3.avi
/experiment/session1/calibration/calib_v1_cam1.avi
/experiment/session1/calibration/calib_v1_cam2.avi
/experiment/session1/calibration/calib_v1_cam3.avi
```

## Configuration

Example config file for 2d/3d tracking:

```toml
model_folder = "/home/tuthill/pierre/DeepLabCut_pierre/pose-tensorflow/models"
model_name = "flywalkMay9"

model_train_iter = 750000

# 3d options

triangulate = true
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
