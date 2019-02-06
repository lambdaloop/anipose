# Setting up Anipose for 3D tracking

First, set up Anipose for 2D tracking by following the instructions [here](start_2d.md).

## Folder structure

Here is the general layout of folders for videos for 3D tracking

```
.../experiment/config.toml
.../experiment/session1/videos-raw/vid_abc_cam1.avi
.../experiment/session1/videos-raw/vid_abc_cam2.avi
.../experiment/session1/videos-raw/vid_abc_cam3.avi
.../experiment/session1/calibration/calib_v1_cam1.avi
.../experiment/session1/calibration/calib_v1_cam2.avi
.../experiment/session1/calibration/calib_v1_cam3.avi
```

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
