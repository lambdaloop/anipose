# Anipose

[![PyPI version](https://badge.fury.io/py/anipose.svg)](https://badge.fury.io/py/anipose)

Anipose is an open-source toolkit for robust, markerless 3D tracking of animal behavior from multiple camera views. It leverages the machine learning toolbox [DeepLabCut](https://github.com/AlexEMG/DeepLabCut) to track keypoints in 2D, then triangulates across camera views to estimate 3D pose.

Anipose consists of four modular components: (1) a 3D calibration module designed to minimize the influence of outliers, (2) a set of filters to resolve errors in 2D detections, (3) a triangulation module that integrates temporal and spatial constraints to obtain accurate 3D trajectories despite 2D tracking errors, and (4) a pipeline for efficiently processing large numbers of videos. These modules are packaged together within Anipose, but the calibration and triangulation functions are also available as an independent library (aniposelib) for use without the full pipeline and applications beyond pose estimation.

The name Anipose comes from **Ani**mal **Pose**, but it also sounds like "any pose".

## Documentation

Up to date documentation may be found at [anipose.org](http://anipose.org) .

## Demos

<p align="center">
<img src="https://raw.githubusercontent.com/lambdaloop/anipose-docs/master/tracking_3cams_full_slower5.gif" width="70%" >
</p>
<p align="center">
Videos of flies by Evyn Dickinson (slowed 5x), <a href=http://faculty.washington.edu/tuthill/>Tuthill Lab</a>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/lambdaloop/anipose-docs/master/hand-demo.gif" width="70%" >
</p>
<p align="center">
Videos of hand by Katie Rupp
</p>



## References

Here are some references for DeepLabCut and other things this project relies upon:
- Mathis et al, 2018, "DeepLabCut: markerless pose estimation of user-defined body parts with deep learning"
- Romero-Ramirez et al, 2018, "Speeded up detection of squared fiducial markers"
