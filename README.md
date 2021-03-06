# Simple Conditional_GAN
Tensorflow implementation of [Conditional Generative Adversarial Network(CGAN)](https://arxiv.org/abs/1411.1784). <br>
This repository is minor change of [Vanilla_GAN](https://github.com/Jisung-Yoon/Vanilla_GAN).
This model can generate MNIST samples with certian labels.

## File discription
- main.py: Main function of implemenation, construct and train the model, generates images
- model.py: CGAN class
- downlad.py: Files for downlading MNIST data sets
- ops.py: Operation functions
- utils.py: Functions dealing with images processing.

## Prerequisites (my environments)
- Python 3.5.2
- Tensorflow > 0.14
- Numpy

## Usage
First, download dataset with:

    $ python download.py mnist

Second, write the main function with configuration you want.

## Results
Result with same latent variables (0 to 100 epochs).<br>
Each number (0 to 9) can be generated as you want by using CGAN.
![result](assets/Result.gif)
