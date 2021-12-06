# cnn-classifier
CNN classifier to classify different radar modulation types

##Description
This package will generate different radar signals and classify them based on modulation type. There are three different modulation types that we use: LFM, Sin wave modulation, and geometric modulation. Later versions may include more complex types. These signals are created with no noise so adding noise will be necessary for more realistic results and deployment. 

##Installation and Usage 
Currently, the only way to use this efficiently is to clone the repo and to move the scripts into your own repo. Later it will be structured so that you can simply use the follow code to install: 
```
python3 -m pip install git+https://github.com/egolfbr/cnn-classifier
```
Then to use 
```
from cnn_radar_classifier import chirp 
from cnn_radar_classifier import cnn
from cnn_radar_classifier import dataset_creation
```
