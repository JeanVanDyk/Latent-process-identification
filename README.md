
<h1 align="center">
  <br>
  Latent process identification
  <br>
</h1>

## Key Features

* Save automatically .npz from amass into .pt files that can be loaded and directly used by the denoising auto encoder.
* Use and train a DAE for implementing missing coordinates in pointcloud.

## Requirements

* Pytorch
* Lightning
* Matplotlib
* Numpy

## Datasets

This code was made to be used on amass datasets, on the format SMPLX. You'll have to dowload the datasets, update the relative links, convert them to .pt files using download_amass_data and only then, the data dowloaded could be used for DAE's training.
