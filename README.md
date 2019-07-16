# KNN-Classification-with-MobileNet

A train-on-the-fly image classifier in the web browser. Using transfer learning, the model makes accurate predictions with minimal training data given. 

# Demo

This project is built with Tensorflow.js, its MobileNet image recognition model and its KNN classification model. The website is built in HTML, Javascript and CSS. For my implementation of training CNN models, please refer to [convolutonal neural network](https://github.com/Jacklu0831/Deep-Learning-Projects/tree/master/Convolutional%20Neural%20Networks).

[insert a video demo here]

# How to Use

Refer to the demo video. Everytime a class button is pressed, a screenshot of the video feed is added to the KNN classifier dataset under the class you pressed. The algorithm constantly predicts the live video stream and outputs its prediction above the buttons.

A few things to make the algorithm predict are:
* Poses
* Faces
* Any objects
* Hand gestures
* Written letters
* Mix all of them up, classes do not have to be in the same category!

# Set Up

Simply clone/fork the repo and build index.html. This version of the project uses script tag so tensorflow.js do not have to be downloaded to your environment. 