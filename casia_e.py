'''
@Author: Jilong Wang
@Date: 2019-01-05 14:44:14
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-09 11:13:41
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


class GaitExtractor:
    def __init__(self, gpuid, det_batch_size=20):
        caffe.set_device(int(gpuid))
        caffe.set_mode_gpu()

        self.det_net, self.op_net, self.seg_net = net_init(det_batch_size=det_batch_size)

    def extract(self, test_set, save_dir):
        frame_result = self.det_net.detect(test_set)
        frame_main_role = find_main_role_in_each_frame(frame_result, self.op_net, test_set)
        start_moving_frame, end_moving_frame = delete_still_frame(frame_main_role)
        frame_main_role = frame_main_role[start_moving_frame:end_moving_frame]
        save_results(frame_main_role, self.op_net, self.seg_net, test_set, save_dir)

class PeopleDetection:
    def __init__(self, modelDeployFile,  modelWeightsFile,  gpuid=0,  threshold=0.60,  img_resize=512, batch_size=25):
        self.img_resize = img_resize
        self.batch_size = batch_size
        self.threshold = threshold
        self.net = None
        self.transformer = None

        self.net = caffe.Net(modelDeployFile, modelWeightsFile, caffe.TEST)

        # detection image preprocessing
        self.net.blobs['data'].reshape(self.batch_size, 3, img_resize, img_resize)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    def detect(self, img_dir):
        print('Processing {}:'.format(img_dir))
        img_dir = img_dir

        # get all image names and sorted by name
        im_names = os.listdir(img_dir)
        im_names = [x for x in im_names if 'jpg' in x]
        im_names.sort(key=lambda x: int(x[5:-4]))

        frame_result = []
        batch_size = self.batch_size
        names = []
        shapes = []
        last_c = 0

        for count, im_name in enumerate(im_names):
            image_file = os.path.join(img_dir, im_name)
            image = caffe.io.load_image(image_file)
            transformed_image = self.transformer.preprocess('data', image)
            self.net.blobs['data'].data[(count - last_c) % batch_size, ...] = transformed_image
            shapes.append(image.shape)
            names.append(im_name)
            if (count + 1 - last_c) % batch_size == 0:
                last_c = count + 1 
                detections = self.net.forward()['detection_out']
                for i in range(batch_size):
                    det_label = detections[0, 0, 500*i:500*(i+1), 1]
                    det_conf = detections[0, 0, 500*i:500*(i+1), 2]
                    det_xmin = detections[0, 0, 500*i:500*(i+1), 3]
                    det_ymin = detections[0, 0, 500*i:500*(i+1), 4]
                    det_xmax = detections[0, 0, 500*i:500*(i+1), 5]
                    det_ymax = detections[0, 0, 500*i:500*(i+1), 6]

                    # print('processing {}'.format(names[i]), end='')
                    result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])
                    frame_result.append((names[i], result, shapes[i]))
                names = []
                shapes = []
                # change batch_size when there is no enough images to fill initial batch_size
                if len(im_names) - count < batch_size:
                    batch_size = len(im_names) - count - 1
            # sys.stdout.flush()
        print('Detection done!')
        sys.stdout.flush
        return frame_result

class OpenPose:
    def __init__(self, modelDeployFile,  modelWeightsFile,  threshold=0.1,  width=-1, height=192.0):
        self.height = height
        self.width = width
        self.threshold = threshold
        self.net = caffe.Net(modelDeployFile, modelWeightsFile, caffe.TEST)

    def get_keypoints(self, op_image,  model='not strict'):
        '''
        @description: get an image then return how many keypoints were detected
        @param {nparray image} 
        @return: number of keypoints
        '''
        # plt.imshow(op_image)
        # plt.show()
        # detection image preprocessing
        if self.width == -1:
            ratio = self.height / op_image.shape[0]
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
        if model == 'strict':
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
        # print(count)
        # sys.stdout.flush()
        return count

class PeopleSegmentation:

    def __init__(self, modelDeployFile, modelParamFile, gpuid = 0, prob_threshold = 0.6, height = 150, width = 100, outLayer='BatchNormBackward60'):
        self.prob_threshold = prob_threshold
        self.height = height
        self.width = width
        self.feature_layer = outLayer
        self.net = caffe.Net(modelDeployFile,  modelParamFile, caffe.TEST)
    
    # Parameter ---------
    # inputImages:  multiple images in one batch (height*width*channel*num)
    # 
    # Return --------
    # out: 4-dim array (height*width*channel*num) containing segmentation results
    def segmentPeople(self, inputImages):
        
        input_height,  input_width = inputImages.shape[:2]
        
        # resize the inputImage to the required size
        resized_inputImages = np.zeros((self.height, self.width, inputImages.shape[2], inputImages.shape[3]),
                                          dtype=np.float32)
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
        out = np.zeros((input_height, input_width, 1, inputImages.shape[3]), dtype=np.uint8)
        for i in range(inputImages.shape[3]):
            out[cv2.resize(result[self.feature_layer][i, 0, :, :],
                           (input_width, input_height)) > self.prob_threshold] = 255
        return np.squeeze(out)



def save_results(frame_main_role, op_net, seg_net, img_dir, save_dir):
    # if having enough frames, then abort the first 5 frame and last 25 frames in order to have intact person
    if len(frame_main_role) == 0:
        print('no gait extracted in this video.')
        sys.exit(0)
    if len(frame_main_role) < 30:
        first_frame = 5
        last_frame = -10
    else:
        first_frame = 5
        last_frame = -20
        
    print("the frist frame is {}, the last frame is {}".format(frame_main_role[first_frame][0][5:-4], frame_main_role[last_frame-1][0][5:-4]))
    for im_name, coord in frame_main_role[first_frame: last_frame]:
        img = cv2.imread(os.path.join(img_dir, im_name), cv2.IMREAD_COLOR)
        xmin, xmax, ymin, ymax = coord

        op_image = img[ymin:ymax, xmin:xmax, :, np.newaxis]
        segResults = seg_net.segmentPeople(op_image)

        cv2.imwrite(os.path.join(save_dir, im_name[:-4] + '_dets.jpg'), segResults)
        print('Saved: ' + os.path.join(save_dir, im_name[:-4] + '_dets.jpg'))
        sys.stdout.flush()

def check_last_1s(frame_main_role, op_net, img_dir):
    for i, pack in enumerate(frame_main_role[-25:]):
        im_name, coord = pack
        img = cv2.imread(os.path.join(img_dir, im_name), cv2.IMREAD_COLOR)
        xmin, xmax, ymin, ymax = coord
        op_image = img[ymin:ymax, xmin:xmax, :]
        if op_net.get_keypoints(op_image, model='strict') !=7 :
            return len(frame_main_role) - 25 + i
    return len(frame_main_role)

def find_max(results, threshold, shape):
    max_area = 0
    Xmin, Xmax, Ymin, Ymax = 0, 0, 0, 0
    for i in range(0, results.shape[0]):
        score = results[i, -2]
        if threshold and score < threshold:
            continue
        label = int(results[i, -1])
        if label != 1:
            continue
        xmin = int(round(results[i, 0] * shape[1]))
        ymin = int(round(results[i, 1] * shape[0]))
        xmax = int(round(results[i, 2] * shape[1]))
        ymax = int(round(results[i, 3] * shape[0]))
        area = (xmax - xmin) * (ymax - ymin)
        if max_area < area:
            max_area = area
            Xmin = xmin
            Xmax = xmax
            Ymin = ymin
            Ymax = ymax
    
    # expand 5% border
    height = Ymax - Ymin
    width = Xmax - Xmin
    h = int(round(height * 0.1))
    w = int(round(width * 0.1))
    Xmin -= w
    Ymin -= h
    Xmax += w
    Ymax += h
    # check border
    Xmin = 0 if Xmin < 0 else Xmin
    Xmax = 0 if Xmax < 0 else Xmax
    Xmin = shape[1] if Xmin > shape[1] else Xmin
    Xmax = shape[1] if Xmax > shape[1] else Xmax
    Ymin = 0 if Ymin < 0 else Ymin
    Ymax = 0 if Ymax < 0 else Ymax
    Ymin = shape[0] if Ymin > shape[0] else Ymin
    Ymax = shape[0] if Ymax > shape[0] else Ymax
    return Xmin, Xmax, Ymin, Ymax

def is_main_role(coord1, coord2):
    '''
    @description: using Euclidean Distance to jugde wether the role is main role or not  
    @param {coord1, coord2} 
    @return: True or False
    '''
    x1_center = (coord1[0] + coord1[1])
    y1_center = (coord1[2] + coord1[3])

    x2_center = (coord2[0] + coord2[1])
    y2_center = (coord2[2] + coord2[3])
    distance = (x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2
    if distance < (1080*0.1)**2 and x1_center != 0 and y1_center != 0:
        return True
    else:
        return False

def find_first_main_role(frame_result, op_net, img_dir):
    '''
    @description: find the first frame where the main role appear
    @param {frame_result:(im_name, result, shape)} 
    @return: the first frame where the main role appear, coord of the main role in this frame
    '''
    print('finding the first frame of the main role.')
    sys.stdout.flush()
    roles = []
    # find main role in each frame
    for i in range(0, len(frame_result), 5):
        im_name, result, shape = frame_result[i]
        xmin, xmax, ymin, ymax = find_max(result, 0.60, shape)
        img = cv2.imread(os.path.join(img_dir, im_name))
        op_image = img[ymin:ymax, xmin:xmax]
        # print("checking frame {}".format(i))
        # print(xmin, xmax, ymin, ymax)
        # sys.stdout.flush()
        if xmin + xmax + ymin + ymax != 0 and op_net.get_keypoints(op_image, model='strict') == 7:
            roles.append((i, [xmin, xmax, ymin, ymax]))

    # find the largest role which definitely is the main role
    max_area = 0.0
    max_index = 0
    main_role_coord = []
    for i, coord in roles:
        xmin, xmax, ymin, ymax = coord
        area = (xmax - xmin) * (ymax - ymin)
        if max_area < area:
            max_area = area
            max_index = i
            main_role_coord = [xmin, xmax, ymin, ymax]

    first_index = roles[0][0]
    # print(first_index, max_index)
    # sys.stdout.flush()
    # find the first frame of the largest role
    if first_index == max_index:
        return max_index, main_role_coord
    search_frame = frame_result[first_index: max_index]
    search_frame.reverse()
    for i in range(len(search_frame)):
        im_name, result, shape = search_frame[i]
        xmin, xmax, ymin, ymax = find_max(result, 0.60, shape)
        if is_main_role([xmin, xmax, ymin, ymax], main_role_coord):
            first_frame = max_index - i
            main_role_coord = [xmin, xmax, ymin, ymax]

    return first_frame, [xmin, xmax, ymin, ymax]

def find_main_role_in_each_frame(frame_result, op_net, img_dir):
    '''
    @description: when more than one person exist in one frame, find out which person is our main role. And track it from the first frame it appears to the frame it disappears.if the frame doesn't contain the main role, abort it
    @param {detection result(im_name, result, shape)} 
    @return main role's coord in each frame(im_name, coord), 
    '''
    frame_main_role = []
    first_frame, main_role_coord = find_first_main_role(frame_result, op_net, img_dir)

    # print('first frame is {}'.format(first_frame))
    # sys.stdout.flush()
    for im_name, result, shape in frame_result[first_frame:]:
        xmin, xmax, ymin, ymax = find_max(result, 0.60, shape)

        if is_main_role([xmin, xmax, ymin, ymax], main_role_coord):
            main_role_coord = [xmin, xmax, ymin, ymax]
            frame_main_role.append((im_name, [xmin, xmax, ymin, ymax]))
        else:
            print(im_name + ' is not main role')
            sys.stdout.flush()
            break

    return frame_main_role
    
def is_moving(coord, still_coord):
    x1_center = (coord[0] + coord[1])
    y1_center = (coord[2] + coord[3])

    x2_center = (still_coord[0] + still_coord[1])
    y2_center = (still_coord[2] + still_coord[3])
    distance = (x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2
    if distance > (1080*0.018)**2:
        return True
    else:
        return False
        
def delete_still_frame(frame_main_role):
    main_role_coord = frame_main_role[0][1]
    for i, pack in enumerate(frame_main_role):
        _, coord = pack
        if is_moving(coord, main_role_coord):
            first_frame = i
            break
    reversed_frame_main_role = frame_main_role[:]
    reversed_frame_main_role.reverse()
    for i, pack in enumerate(frame_main_role):
        _, coord = pack
        if is_moving(coord, main_role_coord):
            last_frame = len(reversed_frame_main_role) - i
            break
    return first_frame, last_frame

def net_init(det_batch_size):
    '''
    @description: load detection & openpose & segementation models
    @param {None} 
    @return: three instances of det_net, op_net, seg_net
    '''
    # load detection model
    modelDeployFile = 'models/detection/deploy.prototxt'
    modelWeightsFile = 'models/detection/coco_refinedet_resnet18_addneg_1024x1024_iter_340000.caffemodel'
    det_net = PeopleDetection(modelDeployFile, modelWeightsFile, img_resize=512, batch_size=det_batch_size, threshold=0.60)

    # load openpose model
    protoFile = "./models/body_25/pose_deploy.prototxt"
    weightsFile = "./models/body_25/pose_iter_584000.caffemodel"
    op_net = OpenPose(protoFile, weightsFile)

    # load segmentation model
    modelDeployFile = './models/segementation/gait_seg.prototxt'
    modelWeightsFile = './models/segementation/gait_seg.caffemodel'
    seg_net = PeopleSegmentation(modelDeployFile, modelWeightsFile, prob_threshold=0.70)
    return det_net, op_net, seg_net

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