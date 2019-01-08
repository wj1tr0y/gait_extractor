<!--
 * @Author: Jilong Wang
 * @Date: 2019-01-08 14:31:09
 * @LastEditors: Jilong Wang
 * @Email: jilong.wang@watrix.ai
 * @LastEditTime: 2019-01-08 14:43:36
 * @Description: file content
 -->

# Extract gait segementation from videos

this project uses modified refineDet model and openpose model.
We provided a 250FPS(on Titan Xp) pedestrian detection RefineDet model(using res18 and reducing refineDet head channels).

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

* open cmake-gui
* create $GaitExtractor_ROOT/build
* press configure button.
* when configuration is done, press generate button
* quit camke

  ```Shell
  cd $GaitExtractor_ROOT
  # Modify Makefile.config according to your Caffe installation.
  # Make sure to include $GaitExtractor_ROOT/python to your PYTHONPATH.
  cd build
  make -j8
  ```

This project based on SSD caffe repo but we modified detection_out_layer and add pixel suffle layer. So if you're using your own caffe, please don't forget copy these source file into your project and re-compile them.

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

if you wanna change the save path, please change them in [gait_extractor.py](https://github.com/wj1tr0y/gait_extractor/blob/master/gait_extractor.py)