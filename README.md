# Anipose

[![PyPI version](https://badge.fury.io/py/anipose.svg)](https://badge.fury.io/py/anipose)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

Anipose is a framework for scalable [DeepLabCut](https://github.com/AlexEMG/DeepLabCut)-based analysis. It supports both 2d and 3d tracking, handles calibration and processing all files within a group of folders.

The name Anipose comes from **Ani**mal **Pose**, but it also sounds like "any pose".

## Getting started

1) Setup DeepLabCut by following instruction [here](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md)
2) Install Anipose through pip: `pip install anipose`

## Why this project?

DeepLabCut is great for training a network to track features in a video, and to run it on a small set of videos.

However, in practice, to accommodate our experiments, we found that we need to write custom code to iterate through folders and videos. Different experimental runs tended to be placed in different folders, and processing this structured data can quickly get overwhelming. This problem is compounded if one wants to do 3D tracking, where many more videos are generated and organization of these is critical for processing data.

Hence, we created Anipose, which places the DeepLabCut feature analysis into a pipeline, organizing the results into folders and autodetecting all the files that need to be processed.

For 2D tracking, Anipose can:
- track all videos in a group of folders
- detect, remove, and interpolate bad tracking
- make videos labeled with the 2D tracked points and lines, and filtered points
- aggregate all the 2D data into one file (easier to analyze further)

For 3D tracking, Anipose can:
- process calibration videos per session (or per experiment, as needed)
- handle triangulation from multiple videos to get 3D points
- generate 3D videos from 3D points
- compute angles in 3D
- aggregate all 3D data and angles into one file (for easier analysis)


## Documentation

- [Setting up Anipose for 2D tracking](./docs/start_2d.md)
- [Setting up Anipose for 3D tracking](./docs/start_3d.md)


## References

Here are some references for DeepLabCut and other things this project relies upon:
```
@article{Mathisetal2018,
    title={DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
    author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
    journal={Nature Neuroscience},
    year={2018},
    url={https://www.nature.com/articles/s41593-018-0209-y}
}

@article{insafutdinov2016eccv,
    title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
    author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele},
    booktitle = {ECCV'16},
    url = {http://arxiv.org/abs/1605.03170}
}

@article{romero-ramirez_speeded_2018,
	title = {Speeded up detection of squared fiducial markers},
	doi = {10.1016/j.imavis.2018.05.004},
	journal = {Image and Vision Computing},
	author = {Romero-Ramirez, Francisco J. and Muñoz-Salinas, Rafael and Medina-Carnicer, Rafael},
	year = {2018},
}

@article{garrido-jurado_generation_2016,
	title = {Generation of fiducial marker dictionaries using {Mixed} {Integer} {Linear} {Programming}},
	doi = {10.1016/j.patcog.2015.09.023},
	journal = {Pattern Recognition},
	author = {Garrido-Jurado, S. and Muñoz-Salinas, R. and Madrid-Cuevas, F.J. and Medina-Carnicer, R.},
	year = {2016},
}
```
