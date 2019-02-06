# Anipose

[![PyPI version](https://badge.fury.io/py/anipose.svg)](https://badge.fury.io/py/anipose)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

Anipose is a framework for scalable [DeepLabCut](https://github.com/AlexEMG/DeepLabCut)-based analysis. It supports both 2d and 3d tracking, handles calibration and processing all files within a group of folders.

The name Anipose comes from **Ani**mal **Pose**, but it also sounds like "any pose".

## Getting started

1) Setup DeepLabCut by following instruction [here](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md)
2) Install Anipose through pip: `pip install anipose`


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
```
