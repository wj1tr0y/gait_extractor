'''
@Author: Jilong Wang
@Date: 2019-01-05 14:44:14
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-08 10:12:21
@Description: In this script, we will load a RefineDet model to detect pedestrian and use openpose to check the integrity of each pedestrian.
'''
import argparse
import os
import sys
import numpy as np
import skimage.io as io
import cv2
import time
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import matplotlib.pyplot as plt
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

def save_results(frame_main_role, img_dir, save_dir):
    # last_frame = check_last_1s(frame_main_role, img_dir)
    # if having enough frames, then abort the first 5 frame and last 25 frames in order to have intact person
    if len(frame_main_role) < 30:
        first_frame = 5
        last_frame = -10
    else:
        first_frame = 5
        last_frame = -25
    for im_name, coord in frame_main_role[first_frame: last_frame]:
        print(im_name, coord)
        img = cv2.imread(os.path.join(img_dir, im_name))
        xmin, xmax, ymin, ymax = coord
        op_image = img[ymin:ymax, xmin:xmax]
        cv2.imwrite(os.path.join(save_dir, im_name[:-4] + '_dets.jpg'), op_image)
        print 'Saved: ' + os.path.join(save_dir, im_name[:-4] + '_dets.jpg')
        sys.stdout.flush()
            
def openpose(op_image, model='not strict'):
    '''
    @description: get an image then return how many keypoints were detected
    @param {nparray image} 
    @return: number of keypoints
    '''
    ratio = 192.0 / op_image.shape[0]
    inWidth = int(round(op_image.shape[1] * ratio))
    inHeight = int(round(op_image.shape[0] * ratio))
    # inWidth = op_image.shape[1]
    # inHeight = op_image.shape[0]
    # Convert image to blob
    netInputSize = (inWidth, inHeight)
    inpBlob = cv2.dnn.blobFromImage(op_image, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    op_net.setInput(inpBlob)

    # Run Inference (forward pass)
    output = op_net.forward()

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
    print(count)
    sys.stdout.flush()
    return count

def detect(test_set, net, transformer,det_batch_size):
    print('Processing {}:'.format(test_set))
    img_dir = test_set
    im_names = os.listdir(img_dir)
    im_names = [x for x in im_names if 'jpg' in x]
    im_names.sort(key=lambda x: int(x[5:-4]))
    frame_result = []
    batch_size = det_batch_size
    names = []
    shapes = []
    last_c = 0
    for count, im_name in enumerate(im_names):
        image_file = os.path.join(img_dir, im_name)
        image = caffe.io.load_image(image_file)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[(count - last_c) % batch_size, ...] = transformed_image
        shapes.append(image.shape)
        names.append(im_name)
        if (count + 1 - last_c) % batch_size == 0:
            last_c = count + 1 
            detections = net.forward()['detection_out']
            for i in range(batch_size):
                det_label = detections[0, 0, 500*i:500*(i+1), 1]
                det_conf = detections[0, 0, 500*i:500*(i+1), 2]
                det_xmin = detections[0, 0, 500*i:500*(i+1), 3]
                det_ymin = detections[0, 0, 500*i:500*(i+1), 4]
                det_xmax = detections[0, 0, 500*i:500*(i+1), 5]
                det_ymax = detections[0, 0, 500*i:500*(i+1), 6]
                try:
                    print('processing {}'.format(names[i]))
                
                    result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])
                    frame_result.append((names[i], result, shapes[i]))
                except:
                    print(batch_size, len(names),len(shapes))
                    sys.exit(0)
            names = []
            shapes = []
            if len(im_names) - count < batch_size:
                batch_size = len(im_names) - count - 1
        sys.stdout.flush()
    return frame_result

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
    h = int(round(height * 0.05))
    w = int(round(width * 0.05))
    Xmin -= w
    Ymin -= h
    Xmax += w
    Ymax += h
    # check border
    Xmin = 0 if Xmin < 0 else Xmin
    Xmax = 0 if Xmax < 0 else Xmax
    Xmin = 1920 if Xmin > 1920 else Xmin
    Xmax = 1920 if Xmax > 1920 else Xmax
    Ymin = 0 if Ymin < 0 else Ymin
    Ymax = 0 if Ymax < 0 else Ymax
    Ymin = 1080 if Ymin > 1080 else Ymin
    Ymax = 1080 if Ymax > 1080 else Ymax
    return Xmin, Xmax, Ymin, Ymax

def is_main_role(coord1, coord2):
    x1_center = (coord1[0] + coord1[1])
    y1_center = (coord1[2] + coord1[3])

    x2_center = (coord2[0] + coord2[1])
    y2_center = (coord2[2] + coord2[3])
    distance = (x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2
    if distance < 10000:

        return True
    else:
        return False

def find_first_main_role(frame_result):
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
        print("checking frame {}".format(i))
        print(xmin, xmax, ymin, ymax)
        sys.stdout.flush()
        if xmin + xmax + ymin + ymax != 0 and openpose(op_image, model='strict') == 7:
            roles.append((i, [xmin, xmax, ymin, ymax]))

    # find the largest role
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
    print(first_index, max_index)
    sys.stdout.flush()
    # find the first frame of the largest role
    if first_index == max_index:
        return max_index, main_role_coord
    search_frame = frame_result[first_index: max_index]
    search_frame.reverse()
    for i in range(len(search_frame)):
        im_name, result, shape = search_frame[i]
        xmin, xmax, ymin, ymax = find_max(result, 0.80, shape)
        if is_main_role([xmin, xmax, ymin, ymax], main_role_coord):
            first_frame = max_index - i
            main_role_coord = [xmin, xmax, ymin, ymax]
    return first_frame, [xmin, xmax, ymin, ymax]

def find_main_role_in_each_frame(frame_result):
    frame_main_role = []
    first_frame, main_role_coord = find_first_main_role(frame_result)
    print('first frame is {}'.format(first_frame))
    sys.stdout.flush()
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

def movement_checker(frame_result):
    slices = []
    first_frame, main_role_coord = find_first_main_role(frame_result)
    for im_name, result, shape in frame_result[first_frame:]:
        xmin, xmax, ymin, ymax = find_max(result, 0.40, shape)
        if is_main_role([xmin, xmax, ymin, ymax], main_role_coord):
            main_role_coord = [xmin, xmax, ymin, ymax]
            frame_main_role.append((im_name, [xmin, xmax, ymin, ymax]))
        else:
            print(im_name + ' is not main role')
            sys.stdout.flush()
            break

    return frame_main_role

def net_init(det_batch_size=20):
    # load detection model
    model_def = 'models/detection/deploy.prototxt'
    model_weights = 'models/detection/coco_refinedet_resnet18_addneg_1024x1024_iter_340000.caffemodel'
    det_net = caffe.Net(model_def, model_weights, caffe.TEST)

    # detection image preprocessing
    img_resize = 512
    det_net.blobs['data'].reshape(det_batch_size, 3, img_resize, img_resize)
    transformer = caffe.io.Transformer({'data': det_net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    # load openpose model
    protoFile = "./models/body_25/pose_deploy.prototxt"
    weightsFile = "./models/body_25/pose_iter_584000.caffemodel"
    op_net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    return op_net, det_net, transformer

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
    caffe.set_device(int(args.gpuid))
    caffe.set_mode_gpu()

    save_dir = args.out_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    test_set = args.test_set
    img_dir = test_set
    if not os.path.exists(test_set):
        print("{} doesn't exists".format(test_set))
        sys.exit(0)

    batch_size = 25
    # initialize openpose and detect net 
    op_net, det_net, transformer = net_init(det_batch_size=batch_size)
    # do detection
    frame_result = detect(test_set, det_net, transformer,det_batch_size=batch_size)
    frame_main_role = find_main_role_in_each_frame(frame_result)
    save_results(frame_main_role, test_set, save_dir)
