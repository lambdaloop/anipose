Setting up Anipose for 3D tracking
##################################

I hear you're interested in tracking animals in three dimensions. Here's
how to set up Anipose for that.

Overall, the set up is as follows: 

- set up Anipose for 2D tracking by
  following the instructions :doc:`here </start2d>`
- setup calibration and folder structure for your experiment
- optionally, specify more post-processing you're interested in 
  (e.g. angles, axis alignment, etc)

Setup calibration
=================

Folder structure
----------------

Here is the general layout of folders for videos for 3D tracking. Within
each session, there is now a ``videos-raw`` folder with the videos to be
tracked and a ``calibration`` folder with the calibration videos.

When looking for a calibration folder, anipose will look first within
the same folder as where ``videos-raw`` is in, then at any folders above
it. This way, a single calibration may be reused across sessions if the
cameras do not move significantly.

.. code-block:: text

    .../experiment/config.toml
    .../experiment/session1/videos-raw/vid_abc_cam1.avi
    .../experiment/session1/videos-raw/vid_abc_cam2.avi
    .../experiment/session1/videos-raw/vid_abc_cam3.avi
    .../experiment/session1/calibration/vid_v1_cam1.avi
    .../experiment/session1/calibration/vid_v1_cam2.avi
    .../experiment/session1/calibration/vid_v1_cam3.avi

Specifying how to parse camera names
------------------------------------

The camera names are identified from the file names using a pattern
specified in the configuration.

This is a regular expression that specifies what component of the file
name is the camera name.

Briefly, the text inside the parentheses is what is parsed as the camera
name, the rest of the pattern can help find a unique position in the
file name.

You can read more about what you can match from the
`python docs <https://docs.python.org/3/library/re.html>`_.

Here is how you might specify a pattern for the example list of
filenames above.

.. code:: yaml

    [triangulation]
    cam_regex = 'cam([0-9])'

(The ``[0-9]`` matches any number between 0 and 9 inclusive.)

Another example, suppose your filenames look like:
``02112019_fly4_0 R2C14 Cam-A str-ccw-0.72 sec.avi``. Here the camera
name is the one capital letter after "Cam-". We can specify this as:

.. code:: yaml

    cam_regex = 'Cam-([A-Z])'

How to record videos for calibration
------------------------------------

In order to perform triangulation well, the cameras need to be
calibrated well. This is crucial. If you are obtaining good tracking but
poor triangulation, the culprit is likely the calibration.

To calibrate your cameras, you need to use a calibration board. Anipose
supports:

-  `Charuco boards <https://docs.opencv.org/3.4.3/df/d4a/tutorial_charuco_detection.html>`_
-  `Checkerboards <https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf>`_
-  `Aruco boards <https://docs.opencv.org/3.4.3/db/da9/tutorial_aruco_board_detection.html>`_

I recommend using a Charuco board or checkerboard. Use an Aruco board
only if absolutely necessary, as I found it can lead to poor
calibration.

To get an image of the calibration board, you may either draw the board
with Anipose (as detailed below), or download some version from online.

Print out the image, place it on a flat board, and collect some synced
videos from your cameras of the checkerboard in different positions (as
you would when collecting behavior).

An example of a good calibration video (ignore the software):
https://vimeo.com/22708033

Some tips for collecting videos for good calibration may be found
`here <https://calib.io/blogs/knowledge-base/calibration-best-practices>`_.

Calibration marker configuration
--------------------------------

Once you have figured out which calibration board you will use, you need
to specify this to anipose.

What to configure: - the type of board (aruco / charuco / checkerboard)
- the size of the board (number squares in X and Y directions) - Length
of marker separation (for aruco) or square side (for charuco or
checkerboard) (triangulation is set to this unit) - Length of marker
side in appropriate unit, in same unit as above - aruco marker
dictionary (number of bits and number of markers in dictionary)

An example configuration:

.. code:: yaml

    [calibration]
    # checkerboard / charuco / aruco
    board_type = "charuco"

    # width and height of grid
    board_size = [6, 6]

    # number of bits in the markers, if aruco/charuco
    board_marker_bits = 4

    # number of markers in dictionary, if aruco/charuco
    board_marker_dict_number = 50

    # length of marker side
    board_marker_length = 3 # mm

    # If aruco, length of marker separation
    # board_marker_separation_length = 1 # mm

    # If charuco or checkerboard, square side length
    board_square_side_length = 4 # mm

Manual verification of calibration pattern detection
----------------------------------------------------

The automatic calibration pattern detection can fail. Removing
incorrectly detected frames will improve calibration accuracy.

What to configure: - Optional boolean (default=false) indicating whether
or not you want to manually verify the detection of the calibration
pattern in each frame (Allows you to throw out bad detections)

To manually verify, add the example below to your config.toml file.

.. code:: yaml

    [manual_verification]
    # true / false
    manually_verify = true

Drawing the calibration board
-----------------------------

If you have specified your calibration marker in the configuration (as
above), you can use anipose to draw it. This can be useful for checking
whether the configuration is correct, or for drawing arbitrary
calibration boards.

.. code:: bash

    anipose draw_calibration

This will output an image named ``calibration.png`` in your project
folder.

Extra features to configure
===========================

Triangulation with cropping
---------------------------

Calibration should always be recorded with the maximum view your camera
offers, for best results. However, behavior may be recorded with cropped
views (e.g. to get a faster frame rate).

Anipose supports this to some extent, but as of yet it is not properly
documented. If you're particularly interested in this feature, please
email Pierre about it.

Configuring the standardized 3D pose
------------------------------------

In order to properly compare across different trials, different animals,
and different setups, it may be useful to standardize 3D coordinates
relative to a common reference frame.

Anipose allows configuration of this by specifying 2 sets of points to
use as axes, and which axes these should be.

The algorithm to determine the axes is as follows: - the first axis is
taken as given - the second axis is orthogonalized with respect to the
first - the third axis is the cross product of the first two axes

An axis is specified as a pair of points, with the axis going from the
first to the second point.

Furthermore, it is often useful to set the zero to a standard reference
point. Anipose allows this too.

An example configuration:

.. code:: yaml

    [triangulation]
    axes = [
        ["x", "L1A", "L3A"],
        ["z", "L1B", "L1A"]
    ]
    reference_point = "L1A"

Computing angle estimates
-------------------------

Although it’s very useful to get 3D tracking estimates, we also need
angle estimates.

However, not all angles make sense, it is up to the user to specify
which angles she cares about.

This may be specified in the config.toml file as follows:

.. code:: yaml

    [angles]
    L1_CF = ["L1A", "L1B", "L1C"]
    L1_FTi = ["L1B", "L1C", "L1D"]
    L1_TiTa = ["L1C", "L1D", "L1E"]

The key above is the ``[angles]`` header, which specifies that whatever
follows is an angle.

Next, each angle is specified by a name on the left, and by a list of 3
joints on the right.
