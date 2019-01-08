# -*- coding: utf-8 -*-
import numpy
import cv2
import time
import os
import sys
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe


class PeopleSegmentation:

    def __init__(self):
        self.pro_threshold = 0.6
        self.height = 150
        self.width = 100
        self.feature_layer = 'BatchNormBackward60'
        self.net = None
    
    # Parameter--------
    # modelDeployFile: deploy file path including file name
    # modelParamFile:  corresponding param file path including file name
    # mode:                   'cpu' or 'gpu'
    # gpuid:                   indicate the gpu index once the mode is set to 'gpu'
    # pro_threshold:      the probability that filters the pixels
    # height:                  the required input image height (like 150)
    # width:                   the required input image width (like 100)
    def loadModel(self, modelDeployFile,  modelParamFile, mode,  gpuid,  pro_threshold,  height,  width, outLayer):
        if mode == 'cpu':
            caffe.set_mode_cpu()
        elif mode == 'gpu' and gpuid >= 0:
            caffe.set_mode_gpu()
            caffe.set_device(gpuid)
        else:
            print 'Unknown caffe mode!! Please select one of the following:\'cpu\' or \'gpu\'.'
            return False
        self.pro_threshold = pro_threshold
        self.height = height
        self.width = width
        self.feature_layer = outLayer
        self.net = caffe.Net(modelDeployFile,  modelParamFile,  caffe.TEST)
        return not (self.net is None)
    
    # Parameter ---------
    # inputImages:  multiple images in one batch (height*width*channel*num)
    # 
    # Return --------
    # out: 4-dim array (height*width*channel*num) containing segmentation results
    def segmentPeople(self, inputImages):
        input_height,  input_width = inputImages.shape[:2]
        
        # resize the inputImage to the required size
        resized_inputImages = numpy.zeros((self.height, self.width, inputImages.shape[2], inputImages.shape[3]),
                                          dtype=numpy.float32)
        for i in range(inputImages.shape[3]):
            # CAUTION: cv2.resize needs size to be specified in the order of (width, height)
            resized_inputImages[:, :, :, i] = cv2.resize(inputImages[:, :, :, i], (self.width,  self.height))

        # convert uint8 to float32 (implemented in the previous step)

        # substract mean
        resized_inputImages[:, :, 0, :] = resized_inputImages[:, :, 0, :] - 104.008
        resized_inputImages[:, :, 1, :] = resized_inputImages[:, :, 1, :] - 116.669
        resized_inputImages[:, :, 2, :] = resized_inputImages[:, :, 2, :] - 122.675

        # permute dimensions (height*width*channel*num->num*channel*height*width)
        resized_inputImages = resized_inputImages.transpose([3, 2, 0, 1])

        self.net.blobs['data'].reshape(resized_inputImages.shape[0], resized_inputImages.shape[1],
                                       resized_inputImages.shape[2], resized_inputImages.shape[3])
        self.net.blobs['data'].data[...] = resized_inputImages
        
        result = self.net.forward()
        out = numpy.zeros((input_height, input_width, 1, inputImages.shape[3]), dtype=numpy.uint8)
        for i in range(inputImages.shape[3]):
            out[cv2.resize(result[self.feature_layer][i, 0, :, :],
                           (input_width, input_height)) > self.pro_threshold] = 255
        return numpy.squeeze(out)


def test_image_without_detection(image_filename):
    modelDeployFile = './models/segementation/gait_seg.prototxt'
    modelParamFile = './models/segementation/gait_seg.caffemodel'

    mode = 'gpu'
    gpuid = 0
    prob_threshold = 0.7
    height = 150
    width = 100
    outLayer = 'BatchNormBackward60'
    peopleSegmentation = PeopleSegmentation()
    if not peopleSegmentation.loadModel(modelDeployFile, modelParamFile, mode, gpuid, prob_threshold, height, width,
                                        outLayer):
        print 'Model load unsuccessful.'
        exit()

    # load image
    img = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    img_batch = img[:, :, :, numpy.newaxis]
    segResults = peopleSegmentation.segmentPeople(img_batch)
    import matplotlib.pyplot as plt
    print(segResults.shape)
    plt.imshow(segResults)
    plt.show()

if __name__ == '__main__':
    # test_image_with_detection('./debug-image/test4.jpg')

    test_image_without_detection('./result-001_scene3_nm_L_090_1_s/frame118_dets.jpg')
