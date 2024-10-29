
<h1 align="center">
  <br>
  Latent process identification
  <br>
</h1>

## Source

The model was originally made according to El Esaway, 2015, Estimation of daily bicycle traffic volumes using sparse data. Then it was modified by adding convolution layer inside our Encoder/Decoder, which suited better our dataset (detecting spatial dependencies, reducing dimensions etc), and by using more advanced recurrent network (GRU/LSTM instead of RNN).

## Key Features

* Save automatically .npz from amass into .pt files that can be loaded and directly used by the denoising auto encoder.
* Use and train a DAE for implementing missing coordinates in pointcloud.

## Requirements

* Pytorch
* Lightning
* Matplotlib
* Numpy

## Datasets

* This code was made to be used on amass datasets, on the format SMPLX. You'll have to dowload the datasets, update the relative links, convert them to .pt files using download_amass_data and only then, the data dowloaded could be used for DAE's training.
* You'll also have to download smplx body model and put it in the smplx folder.
