# Automatic Gate Detection for Autonomous Drone Racing
Author: Killian Dally (4553373)
-
Implementation of gate detection based on "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".
Cloned from a [copy] of the original model [model] built by the authors of the paper. Modified and expanded for the confidential WashingtonOB dataset.


**Instructions**:
-
- Install packages as shown in `requirements.txt` for the mentioned version
- Move the entire content of the `WashingtonOBRace` folder to `data/original` (308 images, 308 masks and 1 CSV file) and delete `instructions.txt`
- Run `main.py`


Notes:
- 
- The `frccn` folder contains the scripts for the networks, data processing and configuration
- `config.py` contains all settings for training and testing. Most settings are default values from the original Faster-RCNN
paper, e.g. anchor box sizes are [128, 256, 512] and the ratios are [1:1, 1:2, 2:1]. However, a maximum number of bounding box proposals from the RPN is set to 20 to reduce run time thanks to the low amount of gates in one image.
- The classifier in `resnet.py` (i.e. second classifier after the RPN) has been simplified given that one object class is present.
- The test script `test.py` has also been simplified to improve run time given that one class only is present
- The folder `mask_cnn (reference)` contains scripts to generate prediction masks for gate detection, solution which proved to be not robust enough. It uses a Unet with a pre-trained encoder from MobileNetV2.


Issues:
-
- Errors can occur if the version of keras and tensorflow are not the ones in the `requirements.txt`. 

References: 
-
[paper]: https://arxiv.org/pdf/1506.01497.pdf
[model]: https://github.com/yhenon/keras-frcnn/
[copy]: https://github.com/you359/Keras-FasterRCNN
[pre-trained weights for ResNet]: https://drive.google.com/file/d/1OmCKlUEYmTjg_jaaN-IQm81eHROU-Gyl/view

[https://arxiv.org/pdf/1506.01497.pdf][paper] 
[https://github.com/yhenon/keras-frcnn/][model] 
[https://github.com/you359/Keras-FasterRCNN][copy] 
[https://drive.google.com/file/d/1OmCKlUEYmTjg_jaaN-IQm81eHROU-Gyl/view][pre-trained weights for ResNet]
