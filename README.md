<!--
 * @Author: Jilong Wang
 * @Date: 2019-01-08 14:31:09
 * @LastEditors: Jilong Wang
 * @Email: jilong.wang@watrix.ai
 * @LastEditTime: 2019-01-08 15:06:20
 * @Description: file content
 -->

# Extract gait segementation from videos

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This project uses modified refineDet model and openpose model.


We provided a 250FPS(on Titan Xp with batch_size=20) pedestrian detection RefineDet model(using res18 and reducing refineDet head channels).

## Installation

1. Get the code. We will call the cloned directory as `$GaitExtractor_ROOT`.

  ```Shell
  git clone https://github.com/wj1tr0y/gait_extractor
  ```

1. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.

  ```Shell
  cd $GaitExtractor_ROOT
  # Modify Makefile.config according to your Caffe installation.
  # Make sure to include $GaitExtractor_ROOT/python to your PYTHONPATH.
  cp Makefile.config.example Makefile.config
  make all -j && make py
  ```

Or you can use camke-gui

* Open cmake-gui
* Create $GaitExtractor_ROOT/build
* Press configure button.
* When configuration is done, press generate button
* Quit camke

  ```Shell
  cd $GaitExtractor_ROOT
  cd build
  make -j8
  ```

This project based on SSD caffe repo but we modified detection_out_layer and add pixel_suffle_layer. So if you're using your own caffe, please don't forget copy these two layer's source files into your project and re-compile them.

## Preparation

1. Download [OpenPose body_25 models](#). By default, we assume the model is stored in `$GaitExtractor_ROOT/models/pose/`.

2. Download [Pedestrian detection models](#). By default, we assume the model is stored in `$GaitExtractor_ROOT/models/detection/`.

3. Download [People_Seg models](#). By default, we assume the model is stored in `$GaitExtractor_ROOT/models/seg/`.

## Usage

``` Shell
  python gait_extractor.py ./VIDEO_FOLDER_PATH/ --gpuid 0
  python gait_extractor.py ./VIDEO_FOLDER_PATH/*.mp4/ --gpuid 0
```

## Save Path

If you wanna change the save path, please change them in [gait_extractor.py](https://github.com/wj1tr0y/gait_extractor/blob/master/gait_extractor.py)