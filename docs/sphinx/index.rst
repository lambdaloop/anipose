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


.. figure:: anipose-tutorial/tracking_3cams_full_slower5.gif
   :align: center

   Videos of flies by Evyn Dickinson (slowed 5x), `Tuthill Lab <http://faculty.washington.edu/tuthill>`_

.. figure:: anipose-tutorial/hand-demo.gif
   :align: center

   Videos of hand by Katie Rupp

Documentation
=============

.. toctree::
   :maxdepth: 2

   params
   start2d
   start3d
   tutorial 
   aniposelib-api

   :caption: Contents:

Collaborators 
=============

- **Pierre Karashchuk**, Neuroscience Graduate Program, University of Washington 
- **Evyn S. Dickinson**, Department of Physiology and Biophysics, University of Washington
- **Katie Rupp**, Department of Physiology and Biophysics, University of Washington
- **Bingni W. Brunton**, Department of Biology, University of Washington
- **John C. Tuthill**, Department of Physiology and Biophysics, University of Washington

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
