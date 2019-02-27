# Tensorflow lite test
Compare tensorflow and tensorflow lite performance on raspberry pi.

## Motivation
The Raspberry Pi is a tiny and affordable computer perfect for IoT but not
that great for deep learning application (just like self driving cars), the
common solution is to remotely run the model, nevertheless this could be
a problem in a task like this, on the other hand tensorflow has an official
solution for running machine learning models on olw specs devices.

So, in this folder we will compare the performance of tensorflow lite on
raspberry pi. If we consider that it has a significantly better performance,
we will choose this option instead of vanilla tensorflow + remote access
in order to avoid input lag.


## Model
We choose inceptionV4 [1] because it is a model that demands a high
computational resource, also there are an official port on tensorflow lite,
so, it would be a great upper bound for self driving car model.


## Instructions
1. Clone [this](https://github.com/kentsommer/keras-inceptionV4) repo, and
   run `evaluate_image.py` file, in order to measure the vanilla tensorflow
   performace.

2. On `lite` folder run the `model.sh` script and then run the `test.py` file,
   in order to measure the tensorflo lite performance.


## References
[1] Szegedy C, Ioffe S, Vanhoucke V, Alemi A. Inception-v4, Inception-ResNet and
    the Impact of Residual Connections on Learning.
    arXiv.org. https://arxiv.org/abs/1602.07261. Published 2016.
