# Image Analysis

[![Test](https://github.com/10XDev/rust-template/actions/workflows/test.yaml/badge.svg)](https://github.com/10XDev/rust-template/actions/workflows/test.yaml)

This repository holds rust implementations of image analysis algorithms used in Spaceranger.
Currently, this includes implementations of Visium HD fiducial detection and basic image transformations.

## Fiducial Detection and Registration
Fiducial code for Visium HD. Contains the fiducial detection, decoding and registration algorithms used in Spaceranger.
An outline of the fiducial detection algorithm is described [here][]. 

[here]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Calvet_Detection_and_Accurate_CVPR_2016_paper.pdf
