# Skateboard Trick Detector/Classifier/Estimator/Predictor/Identifier

On creating a neural network based approach for identifying skateboard tricks from video. Classifying the different ways a skateboard rotates is a difficult task for anyone not familiar with the activity. From a computer vision perspective, skateboard tricks happen quickly, which means the implementation needs to capture the fine details over a few hundred milliseconds that distinguish a trick. This requires a model that can perform real-time predictions on high resolution data. 

## Dataset

* There isn't one.
* High framerate to capture more details of the skateboard motion
* High resolution to better detect the skateboard

## Architectures

* Action detection
  * Temporal Bounds: Findind start and end times of trick
  * RCNN, LSTM, etc: Some kind of 'memory' for classifying motion
* Real-time performance
