# Anipose

[![PyPI version](https://badge.fury.io/py/anipose.svg)](https://badge.fury.io/py/anipose)

Anipose is an open-source toolkit for robust, markerless 3D pose estimation of animal behavior from multiple camera views. It leverages the machine learning toolbox [DeepLabCut](https://github.com/AlexEMG/DeepLabCut) to track keypoints in 2D, then triangulates across camera views to estimate 3D pose.

Check out the [Anipose paper](https://www.cell.com/cell-reports/fulltext/S2211-1247(21)01179-7) for more information.

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
