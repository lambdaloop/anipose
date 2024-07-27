#!/usr/bin/env bash

rm -rv dist build *.egg-info
python setup.py sdist bdist_wheel
# python3 setup.py bdist_wheel
twine upload dist/*
