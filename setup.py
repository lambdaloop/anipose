import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="anipose",
    version="1.0.1",
    author="Pierre Karashchuk",
    author_email="krchtchk@gmail.com",
    description="Framework for scalable DeepLabCut based analysis including 3D tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lambdaloop/anipose",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
    entry_points={
        'console_scripts': ['anipose=anipose.anipose:cli']
    },
    install_requires=[
        'deeplabcut>=2.0.4.1',
        'aniposelib>=0.4.3',
        'opencv-python<=4.6.0.66',
        'opencv-contrib-python<=4.6.0.66',
        'toml',
        'numpy',
        'scipy',
        'pandas',
        'tqdm',
        'click',
        'scikit-video',
        'flask', 'flask-compress', 'flask-ipban'
    ],
    extras_require={
        'viz':  ["mayavi"]
    }

)
