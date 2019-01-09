'''
@Author: Jilong Wang
@Date: 2019-01-05 14:44:14
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-09 17:54:13
@Description: In this script, we will load a RefineDet model to detect pedestrian and use openpose to check the integrity of each pedestrian.
finally, we will use a small segmentation model to seg person in each frame then save the result.
'''

from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import skimage.io as io
import cv2
import time
import shutil
import matplotlib.pyplot as plt
os.environ['GLOG_minloglevel'] = '3'
# Make sure that caffe is on the python path:x
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import matplotlib.pyplot as plt
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2


class OpenPose:
    def __init__(self, modelDeployFile,  modelWeightsFile,  threshold=0.1,  width=-1, height=192.0):
        self.height = height
        self.width = width
        self.threshold = threshold
        self.net = caffe.Net(modelDeployFile, modelWeightsFile, caffe.TEST)

    def get_keypoints(self, im_name, mode='not strict'):
        '''
        @description: get an image then return how many keypoints were detected
        @param {nparray image} 
        @return: number of keypoints
        '''
        # detection image preprocessing
        op_image = cv2.imread(im_name)
        op_image = cv2.cvtColor(op_image, cv2.COLOR_BGR2RGB)
        if self.width == -1:
            ratio = 1
            inWidth = int(round(op_image.shape[1] * ratio))
            inHeight = int(round(op_image.shape[0] * ratio))
        else:
            inWidth = self.width
            inHeight = self.height

        self.net.blobs['image'].reshape(1, 3, inHeight, inWidth)
        transformer = caffe.io.Transformer({'data': self.net.blobs['image'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([0, 0, 0]))  # mean pixel
        transformer.set_raw_scale('data', 1/255.0)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
        transformed_image = transformer.preprocess('data', op_image)
        self.net.blobs['image'].data[0, ...] = transformed_image
        output = self.net.forward()['net_output']

        scaleX = float(op_image.shape[1]) / output.shape[3]
        scaleY = float(op_image.shape[0]) / output.shape[2]

        # Empty list to store the detected keypoints
        points = []
        # Confidence treshold 
        threshold = 0.1
        for i in range(25):
            # Obtain probability map
            probMap = output[0, i, :, :]
            
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = scaleX * point[0]
            y = scaleY * point[1]

            if prob > threshold : 
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else :
                points.append(None)
        count = 0
        if mode == 'strict':
            if points[1] is not None: # Neck
                count += 1
            if points[10] and points[13]: # Knee
                count += 1
            if points[11] and points[14]: # Ankle
                count += 1
            if points[19] and points[22]: # BigToe
                count += 1
            else:
                count -= 1
            if points[21] and points[24]: # Heel
                count += 1
            else:
                count -= 1
            if points[9] and points[8] and points[12]: # Hip
                count += 1
            if points[2] and points[5]: # double Shoulder
                count += 1
            else:
                count -= 1
        else:
            if points[1] is not None: # Neck
                count += 1
            if points[10] and points[13]: # Knee
                count += 1
            if points[11] or points[14]: # Ankle
                count += 1
            if points[19] or points[22]: # BigToe
                count += 1
            if points[21] and points[24]: # Heel
                count += 1
            if points[9] or points[8] or points[12]: # Hip
                count += 1
            if points[2] and points[5]: # double Shoulder
                count += 1
            else:
                count -= 1
        return count

class GaitExtractor:
    def __init__(self, gpuid, det_batch_size=20):
        caffe.set_device(int(gpuid))
        caffe.set_mode_gpu()

        self.op_net = net_init(det_batch_size=det_batch_size)

    def extract(self, img_dir, save_dir):
        frame_result = []
        print('Processing {}:'.format(img_dir))

        # get all image names and sorted by name
        im_names = os.listdir(img_dir)
        im_names = [x for x in im_names if 'jpg' in x]
        im_names.sort(key=lambda x: int(x[5:-4]))

        frame_result = []
        for im_name in im_names:
            frame_result.append((im_name, self.op_net.get_keypoints(os.path.join(img_dir, im_name), mode='strict')))


        first_frame = find_first_role(frame_result)
        last_frame = len(frame_result) - find_first_role(reversed(frame_result))
        save_results(frame_result, first_frame, last_frame, img_dir, save_dir)

def save_results(frame_result, first_frame, last_frame, img_dir, save_dir):
    # if having enough frames, then abort the first 5 frame and last 25 frames in order to have intact person
    if last_frame - first_frame < 15:
        return

    print("the frist frame is {}, the last frame is {}".format(frame_result[first_frame][0][5:-4], frame_result[last_frame-1][0][5:-4]))
    for im_name, _ in frame_result[first_frame: last_frame]:
        img = cv2.imread(os.path.join(img_dir, im_name), cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(save_dir, im_name[:-4] + '_dets.jpg'), img)
        print('Saved: ' + os.path.join(save_dir, im_name[:-4] + '_dets.jpg'))

def find_first_role(frame_result):
    for i, pack in enumerate(frame_result):
        _, keypoints = pack
        if keypoints == 7:
            return i + 5
def net_init(det_batch_size):
    '''
    @description: load detection & openpose & segementation models
    @param {None} 
    @return: three instances of det_net, op_net, seg_net
    '''
    # load openpose model
    protoFile = "./models/body_25/pose_deploy.prototxt"
    weightsFile = "./models/body_25/pose_iter_584000.caffemodel"
    op_net = OpenPose(protoFile, weightsFile)
    return op_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "run test and get all result")
    parser.add_argument("--gpuid",
        help = "The gpu chosen to run the model.", required=True)
    parser.add_argument("--out-dir",
        help = "The output directory where we store the result.", required=True)
    parser.add_argument("--test-set", 
        help = "which sets your wanna run test.", required=True)

    args = parser.parse_args()
    # gpu preparation
    assert len(args.gpuid) == 1, "You only need to choose one gpu. But {} gpus are chosen.".format(args.gpuid)
    
    save_dir = args.out_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_dir = args.test_set
    if not os.path.exists(img_dir):
        print("{} doesn't exists".format(img_dir))
        sys.exit(0)

    gait = GaitExtractor(args.gpuid)
    gait.extract(img_dir, save_dir)
