# Face detection & Emotion recognition
------------------------------------------------
## Table of Contents

* [Pre-requisites](#pre-requisites)
* [Quick Start](#quick-start)
* [Usage](#usage)
  * [Dataset](#dataset)
  * [Face Detection](#face-detection)
  * [Pre-trained Model](#pre-trained-model)
* [Sample Outputs](#sample-outputs)
* [References](#references)


## Pre-requisites

* argparse
    > pip install argparse
* Keras
    > pip install keras
* opencv-python
    > pip install opencv-python
* opencv-contrib-python
    > pip install opencv-contrib-python
* Numpy
    > pip install numpy
* facenet-pytorch
    > pip install facenet-pytorch

## Quick Start

* Clone this repository: $ git clone https://github.com/jaehwan-AI/face_detect

* Run the demo:

>**image input**
```bash
$ python detect_demo.py --image data/image/image.jpg
```

>**video input**
```bash
$ python detect_demo.py --video data/video/video.mp4
```

>**webcam**
```bash
$ python detect_demo.py --src 0
```

## Usage

### Dataset

This model trained with FER2013 dataset. [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
The data consists of 48x48 pixel grayscale images of faces. The training set consists of 28,709 examples. The public test set used for the validation consists of 3,589 examples. The dataset have pictures based on the emotion shown in the facial expression in to one of seven categories.

### Face Detection

We used MTCNN as a facial recognition technology to analyze emotions. MTCNN uses image pyramids by resizing images entered on different scales to recognize faces of different sizes in the images.

### Pre-trained Model

In order to inference the model, we used pre-learned weights using XCEPTION developed from Google(2017).

<img src="xception.PNG" width="60%">

## Sample Outputs

sample image:

<img src="sample/sample1.jpg" width="60%">

sample video:

<img src="sample/sample2.gif" width="60%">

sample webcam:

<img src="sample/sample3.gif" width="60%">

## References

1. Tim Esler's facenet-pytorch repo: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)

1. Octavio Arriaga's pre-trained model repo: [https://github.com/oarriaga/face_classification](https://github.com/oarriaga/face_classification)

1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. _Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks_, IEEE Signal Processing Letters, 2016. [PDF](https://arxiv.org/abs/1604.02878)

1. F. Chollet. _Xception: Deep Learning with Depthwise Separable Convolutions_, 2017. [PDF](https://arxiv.org/abs/1610.02357)
