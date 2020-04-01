Anipose
#######

|PyPI version| |DOI| |License: LGPL v3|

Anipose is a framework for scalable
`DeepLabCut <https://github.com/AlexEMG/DeepLabCut>`_-based analysis.
It supports both 2d and 3d tracking, handles calibration and processing
all files within a group of folders.

The name Anipose comes from **Ani**\ mal **Pose**, but it also sounds
like "any pose".

Getting started
===============

1) Setup DeepLabCut by following the instructions
   `here <https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md>`_
2) Install Anipose through pip: ``pip install anipose``

Documentation
=============

-  :doc:`Setting up Anipose for 2D tracking </start2d>`
-  :doc:`Setting up Anipose for 3D tracking </start3d>`

Demos
=====

-  :doc:`Anipose Tutorial </tutorial>`

.. figure:: anipose-tutorial/tracking_3cams_full_slower5.gif
   :scale: 70 %
   :align: center

   Videos of flies by Evyn Dickinson (slowed 5x), `Tuthill Lab <http://faculty.washington.edu/tuthill>`_

.. figure:: anipose-tutorial/hand-demo.gif
   :scale: 70 %
   :align: center

   Videos of hand by Katie Rupp

Why this project?
=================

DeepLabCut is great for training a network to track features in a video,
and to run it on a small set of videos.

However, in practice, to accommodate our experiments, we found that we
need to write custom code to iterate through folders and videos.
Different experimental runs tended to be placed in different folders,
and processing this structured data can quickly get overwhelming. This
problem is compounded if one wants to do 3D tracking, where many more
videos are generated and organization of these is critical for
processing data.

Hence, we created Anipose, which places the DeepLabCut feature analysis
into a pipeline, organizing the results into folders and autodetecting
all the files that need to be processed.

For 2D tracking, Anipose can:

- track all videos in a group of folders 
- detect, remove, and interpolate bad tracking 
- make videos labeled with the 2D tracked points and lines, and filtered
  points 
- aggregate all the 2D data into one file (easier to analyze further)

For 3D tracking, Anipose can: 

- process calibration videos per session (or per experiment, as needed)
- handle triangulation from multiple videos to get 3D points 
- generate 3D videos from 3D points 
- compute angles in 3D
- aggregate all 3D data and angles into one file (for easier analysis)

References
==========

Here are some references for DeepLabCut and other things this project
relies upon: 

- Mathis et al, 2018, "DeepLabCut: markerless pose
  estimation of user-defined body parts with deep learning" 
- Insafutdinov et al, 2016, "DeeperCut: A Deeper, Stronger, and Faster
  Multi-Person Pose Estimation Model" 
- Romero-Ramirez et al, 2018, "Speeded up
  detection of squared fiducial markers" 
- Garrido-Jurado et al, 2016, "Generation of fiducial marker dictionaries
  using Mixed Integer Linear Programming"

.. |PyPI version| image:: https://badge.fury.io/py/anipose.svg
   :target: https://badge.fury.io/py/anipose
.. |DOI| image:: https://zenodo.org/badge/165723389.svg
   :target: https://zenodo.org/badge/latestdoi/165723389
.. |License: LGPL v3| image:: https://img.shields.io/badge/License-LGPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/lgpl-3.0

User Guide
==========

.. toctree::
   :maxdepth: 2

   params
   start2d
   start3d
   tutorial 
   aniposelib-api

   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
