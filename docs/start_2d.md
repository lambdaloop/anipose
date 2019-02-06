# Setting up Anipose for 2D tracking

So you want to setup Anipose for 2D tracking? Well, you've come to the right place!

It's really quite simple, all you need to do are the following:

1) Train a network to label your data [using DeepLabCut](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/UseOverviewGuide.md)
2) Setup your folder structure for the experiment in the appropriate format
3) Create a `config.toml` file with the parameters for your experiment

## Folder structure

Anipose tries to follow the organization that people tend to come to
naturally for organizing behavioral videos, with one key modification.

The modification is that the *final folder* should be `videos-raw` (the name may be configurable, but it should be the same for each folder). This allows Anipose to know these are the input videos, and to create further folders with the processed data (e.g. `pose-2d`, `pose-2d-filtered`, `videos-labeled`).


Here is the general layout of files for videos for 2D tracking

```
experiment/config.toml
experiment/session1/videos-raw/vid1.avi
experiment/session1/videos-raw/vid2.avi
experiment/session2/videos-raw/vid1.avi
experiment/session2/videos-raw/vid2.avi
```

There is one main experiment folder, and some subfolders under that.
The names for the experiment and session folders can be whatever you like.

Furthermore, the nesting can be arbitrarily large. Thus, an equally valid structure could be (here with nesting of 2 folders instead of 1 as above, note the arbitrary names as well):
```
flypose/config.toml
flypose/session-2019-02-06/fly1/videos-raw/vid-2019-02-06-01.avi
flypose/session-2019-02-06/fly2/videos-raw/vid-2019-02-06-02.avi
flypose/session-2019-02-07/fly1/videos-raw/vid-2019-02-07-01.avi
```


## Configuration

Got your folder structure setup? Great!
Let's setup the configuration!

Anipose uses the [TOML](https://github.com/toml-lang/toml) configuration format, as it is simple and can be read/written easily from Python and Matlab. Consider reading through the [specification](https://github.com/toml-lang/toml) to get a sense of it.


Here is an annotated config file for 2D tracking:

```toml
# Project name
project = "flypose"

# Change this to match deeplabcut folder (one with trained network)
model_folder = '/Data/Videos/DLC_Analysis/Running-Brandon-2019-01-29'

# How many folders are nested in structure?
nesting = 1

[pipeline]
videos_raw = "videos-raw" # change this if you'd like to change name of "videos-raw" folder

# Settings for a threshold filter
# Removes data outside threshold (probably errors in tracking), and interpolates
[filter]
enabled = true
medfilt = 13 # length of median filter
offset_threshold = 25 # offset from median filter to count as jump
score_threshold = 0.8 # score below which to count as bad
spline = true # interpolate using cubic spline instead of linear

# labeling scheme...specify lines that you want to draw for visualizing labels in videos
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

  - **videos-raw** = input videos
  - **pose-2d** = 2d tracking for each of the input videos
  - **pose-2d-filtered** = filtered version of 2d tracking
  - **videos-labeled** = videos labeled with the 2d tracking
  - **videos-labeled-filtered** = videos labeled with the filtered 2d tracking
  - **calibration** = camera parameters obtained from 3d calibration, along with calibration videos
  - **pose-3d** = 3d tracking for each group of input videos
  - **angles** = computed angles from 3d tracking
  - **videos-3d** = 3d videos generated from 3d tracking

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
