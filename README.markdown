# BlazeFace in Python

BlazeFace is a fast, light-weight face detector from Google Research. [Read more](https://sites.google.com/view/perception-cv4arvr/blazeface), [Paper on arXiv](https://arxiv.org/abs/1907.05047)

A pretrained model is available as part of Google's [MediaPipe](https://github.com/google/mediapipe/blob/master/mediapipe/docs/face_detection_mobile_gpu.md) framework.

![](https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/docs/images/realtime_face_detection.gif)

Besides a bounding box, BlazeFace also predicts 6 keypoints for face landmarks (2x eyes, 2x ears, nose, mouth).

Because BlazeFace is designed for use on mobile devices, the pretrained model is in TFLite format. However, I wanted to use it from PyTorch and so I converted it.

> **NOTE:** The MediaPipe model is slightly different from the model described in the BlazeFace paper. It uses depthwise convolutions with a 3x3 kernel, not 5x5. And it only uses "single" BlazeBlocks, not "double" ones.

The BlazePaper paper mentions that there are two versions of the model, one for the front-facing camera and one for the back-facing camera. This repo includes only the frontal camera model, as that is the only one I was able to find an official trained version for. The difference between the two models is the dataset they were trained on. As the paper says,

> For the frontal camera model, only faces that occupy more than 20% of the image area were considered due to the intended use case (the threshold for the rear-facing camera model was 5%).

This means the included model will not be able to detect faces that are relatively small. It's really intended for selfies, not for general-purpose face detection.

## Inside this repo

Essential files:

- **blazeface.py**: defines the `BlazeFace` class that does all the work

- **blazeface.pth**: the weights for the trained model

- **anchors.npy**: lookup table with anchor boxes

Notebooks:

- **Anchors.ipynb**: creates anchor boxes and saves them as a binary file (anchors.npy)

- **Convert.ipynb**: loads the weights from the TFLite model and converts them to PyTorch format (blazeface.pth)

- **Inference.ipynb**: shows how to use the `BlazeFace` class to make face detections

## Detections

Each face detection is a PyTorch `Tensor` consisting of 17 numbers:

- The first 4 numbers describe the bounding box corners: 
    - `ymin, xmin, ymax, xmax`
    - These are normalized coordinates (between 0 and 1).

- The next 12 numbers are the x,y-coordinates of the 6 facial landmark keypoints:
    - `right_eye_x, right_eye_y`
    - `left_eye_x, left_eye_y`
    - `nose_x, nose_y`
    - `mouth_x, mouth_y`
    - `right_ear_x, right_ear_y`
    - `left_ear_x, left_ear_y`
    - Tip: these labeled as seen from the perspective of the person, so their right is your left.

- The final number is the confidence score that this detection really is a face.

## Image credits

Included for testing are the following images:

- **1face.png**. Fei Fei Li by [ITU Pictures](https://www.flickr.com/photos/itupictures/35011409612/), CC BY 2.0

- **3faces.png**. Geoffrey Hinton, Yoshua Bengio, Yann Lecun. Found at [AIBuilders](https://aibuilders.ai/le-prix-turing-recompense-trois-pionniers-de-lintelligence-artificielle-yann-lecun-yoshua-bengio-et-geoffrey-hinton/)

- **4faces.png** from Andrew Ng’s Facebook page / [KDnuggets](https://www.kdnuggets.com/2015/03/talking-machine-deep-learning-gurus-p1.html)

These images were scaled down to 128x128 pixels as that is the expected input size of the model.
