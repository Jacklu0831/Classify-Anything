# KNN-Classification-with-MobileNet

A train-on-the-fly real-time webcam classifier in the web browser I made when learning [**TensorFlow.js**](https://www.tensorflow.org/js) for model deployment. Built with TF.js, HTML, JS, and CSS.

Used **Transfer Learning** with [**MobileNet**](https://arxiv.org/abs/1704.04861) as the base model and a **KNN Classifier** on top of the extracted features, the model inference makes accurate predictions with minimal training data given. MobileNet is the best model for live video stream due to being light weight (**Depth Separable Convolution**) but it does not have the best accuracy in object detection/classification. This con is mitigated by limiting the scope of the problem to a few classes/types of images with the trainable KNN classifier. 

# Demo

<p align="center">
  <a href="http://www.youtube.com/watch?v=rnbaTHqMwyg"><b>Watch the 3am Video Demo</b></a>
  <br>
  <a href="http://www.youtube.com/watch?v=rnbaTHqMwyg"><img src="http://img.youtube.com/vi/rnbaTHqMwyg/0.jpg" title="meme" alt="Video Demo"></a>
</p>

In the video demo, I included 5 classes/types of images. I first clicked on the 5 classes buttons to capture images for training the KNN classifer, then I evaluated the trained model by examining the predictions it makes as I switch between different classes. Focus on how the prediction the model makes changes as I switch different classes by taking off each accessorie. Note that the classes of images can be different poses, expressions, objects, environments, hand gestures, written letters... sky ain't the limit.

**CLASS 1**&emsp;My head\
**CLASS 2**&emsp;My head with blue ray glasses\
**CLASS 3**&emsp;My head with blue ray glasses and a hat\
**CLASS 4**&emsp;My head with blue ray glasses and a hat and a headphone\
**CLASS 5**&emsp;My head with blue ray glasses and a hat and a headphone and a big bottle of water

# How to Use

Refer to the demo video. Everytime a class button is pressed, a screenshot of the video feed is added to the KNN classifier dataset under the class you pressed. The algorithm constantly predicts the live video stream and outputs its prediction above the buttons.

# Set Up

Simply clone/fork the repo and build index.html. This version of the project uses script tag so tensorflow.js do not have to be downloaded to your environment. 

# Paper
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
