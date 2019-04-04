<!--
 * @Author: Jilong Wang
 * @LastEditors: Jilong Wang
 * @Email: jilong.wang@watrix.ai
 * @Description: file content
 * @Date: 2019-02-20 12:52:44
 * @LastEditTime: 2019-02-20 12:54:08
 -->

# Installation

``` shell
  mkdir build
  cd build
  cmake ..
  make -j8
```

# Usage

``` shell
  python ped_test.py testvideos/ --gpuid 0
  or
  python ped_test.py testvideo.mp4 --gpuid 0
```