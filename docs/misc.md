## Output folders

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
