Anipose Configuration File Parameters
#####################################

There are many parameters in the Anipose ``config.toml`` file that can be changed according
to your objectives and preferences when analyzing your data, so it is important to 
familiarize youself with them. These parameters pertain to
project setup, camera calibration, filtering, triangulation, and angle calculation. 
Descriptions of each of these parameters are provided below.

Parameters for Setting up the Project
=====================================
| **project:** Name of the Anipose project (do not edit).
| **model_folder:** This should specify the path to the DeepLabCut project folder. If the 
  DeepLabCut folder is moved, the path should be changed accordingly. 
| **nesting:** Specifies the number of folders that are nested in structure.
| **video_extension:** Specifies the file extension of the calibration videos.

Parameters for Calibration
==========================
| **board_type:** Specifies the type of board used for calibration (``"checkerboard"``, ``"charuco"``, or ``"aruco"``).
| **board_size:** Width and height of grid (mm).
| **board_marker_bits:** If ArUco or ChArUco boards are used for calibration, specifies the number of bits in the markers.
| **board_marker_dict_number:** If ArUco or ChArUco boards are used for calibration, specifies the number of markers in the dictionary.
| **board_marker_length:** Specifies the length of marker side (mm).
| **board_marker_separation_length:** If ArUco boards are used for calibration, specifies the length of marker separation (mm).
| **board_square_side_length:** If chArUco or checkerboards are used for calibration, specifies the, square side length (mm).
| **animal_calibration:** Set to ``true`` if an animal was used for camera calibration.
| **fisheye:** Set to ``true`` if the videos were taken using fisheye lens. Default set to ``false``.

Parameters for 2D Filtering 
===========================
..
	Settings for a threshold filter
	Removes data outside threshold (probably errors in tracking), and interpolates

| **enabled:** If ``true``, enables 2D filtering of the data. 
| **type:** Specifies the type of filter to use on the data (``"viterbi"`` or ``"autoencoder"``).
| **score_threshold:** Score below which labels are determined erroneous. 
| **n_back:** 
| **medfilt:** Specifies the length of the median filter.
| **offset_threshold:** Specifies the offset from median filter to count as a jump.
| **spline:** If ``true``, interpolates using cubic spline instead of linear interpolation. 
| **autoencoder_path:** If the filter type is ``"autoencoder"``, specifies the path to the 
  autoencoder file relative to ``config.toml``.
| **multiprocess:** 

Parameters for Labeling
=======================
| **scheme:** Defines the labeling scheme for the keypoints.

Parameters for Triangulation
============================
| **triangulate:** = Enables triangulation if ``true``. Default is set to ``true``.
| **cam_regex:** = Regular expression to specify the camera names (ex: ``cam_regex`` = ``'_([A-Z])$'``).
| **optim:** If ``true``, enables optimization and applies 3D filters. 
| **constraints:** Pairs of joints to impose smoothness and spatial constraints between. 
| **scale_smooth:**  Strength of enforcement of the smoothing constraints.
| **scale_length:**  Strength of enforcement of the spatial constraints.
| **score_threshold:** Score below which labels are determined erroneous for the 3D filters.

Parameters for Angle Calculation
================================
| **angle:** Specifies a list of three keypoints to compute the angle between. There can
  any number of ``angle`` parameters, and the variable name of each one should 
  correspond to the name of the angle (decided by you). When the angles are written to
  a ``csv`` file, the column headers correspond to the names you specified for each list of three 
  keypoints. There is also the option to compute the angle for one of three types of rotations associated
  with the three keypoints. This can be done by specifying the string ``'flex'``, ``'axis'``, 
  or ``'cross-axis'`` as the first element in the list of angles. The following three 
  elements in the list are still the three keypoints. If no rotation type is specified, the default
  rotation type is ``'flex'``, the flexion-exension angle associated with the three keypoints.
