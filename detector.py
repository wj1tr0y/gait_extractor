'''
@Author: Jilong Wang
@Date: 2019-01-05 14:44:14
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-04-04 13:41:02
@Description: In this script, we will load a RefineDet model to detect pedestriancd .
'''

from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import skimage.io as io
import cv2
import time
import json
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from models import SegmentModel


os.environ['GLOG_minloglevel'] = '3'
# Make sure that caffe is on the python path:x
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import matplotlib.pyplot as plt
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2


class PeopleDetection:
    def __init__(self, modelDeployFile,  modelWeightsFile, seg_model, gpuid=0,  threshold=0.60,  img_resize=512, batch_size=25):
        caffe.set_device(int(gpuid))
        caffe.set_mode_gpu()
        self.img_resize = img_resize
        self.batch_size = batch_size
        self.threshold = threshold
        self.net = None
        self.transformer = None
        self.img = dict()
        self.net = caffe.Net(modelDeployFile, modelWeightsFile, caffe.TEST)

        # detection image preprocessing
        self.net.blobs['data'].reshape(self.batch_size, 3, img_resize, img_resize)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
        self.transform_seg = transforms.Normalize(mean=(104.008, 116.669, 122.675), std=(1, 1, 1))
        self.seg_model = seg_model
        
    def detect(self, img_dir):
        print('Processing {}:'.format(img_dir))

        # get all image names and sorted by name
        im_names = os.listdir(img_dir)
        im_names = [x for x in im_names if 'jpg' in x]

        frame_result = []
        batch_size = self.batch_size
        names = []
        last_c = 0

        for count, im_name in enumerate(im_names):
            image_file = os.path.join(img_dir, im_name)
            image = caffe.io.load_image(image_file)
            self.img[im_name] = image
            transformed_image = self.transformer.preprocess('data', image)
            self.net.blobs['data'].data[(count - last_c) % batch_size, ...] = transformed_image

            names.append(im_name)
            if (count + 1 - last_c) % batch_size == 0:
                last_c = count + 1
                import time
                start = time.time()
                detections = self.net.forward()['detection_out']
                print(time.time()-start)
                for i in range(batch_size):
                    det_label = detections[0, 0, 500*i:500*(i+1), 1]
                    det_conf = detections[0, 0, 500*i:500*(i+1), 2]
                    det_xmin = detections[0, 0, 500*i:500*(i+1), 3]
                    det_ymin = detections[0, 0, 500*i:500*(i+1), 4]
                    det_xmax = detections[0, 0, 500*i:500*(i+1), 5]
                    det_ymax = detections[0, 0, 500*i:500*(i+1), 6]
 
                    # print('processing {}'.format(names[i]), end='')
                    result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])
                    frame_result.append((names[i], result))
                names = []

                # change batch_size when there is no enough images to fill initial batch_size
                if len(im_names) - count <= batch_size:
                    batch_size = len(im_names) - count - 1
                    
            # sys.stdout.flush()
        print('Detection done! Total:{} frames'.format(len(frame_result)))
        self.frame_result = frame_result

    def get_seg(self, img):
        input_width = 100
        input_height = 150
        best_threshold = 0.8

        # set segmentation input
        im_crop_res = cv2.resize(img, (input_width, input_height))
        im_crop_res = torch.from_numpy(im_crop_res.transpose((2, 0, 1)))
        im_crop_res = self.transform_seg(im_crop_res.float())
        im_input = torch.zeros(1, 3, input_height, input_width)
        im_input[0] = im_crop_res
        self.seg_model.set_test_input(im_input)
        
        pred = self.seg_model.forward().cpu()
        pred = pred.data.numpy()
        pred_img = pred[0].squeeze()
        
        pred_img = cv2.resize(pred_img, (img.shape[1], img.shape[0]))

        pred_img[pred_img > best_threshold] = 255
        pred_img[pred_img < best_threshold] = 0
        img[:, :, 2][pred_img > best_threshold] = 255

        return img, pred_img



    def save_results(self, save_dir):
        anns = []
        for im_name, results in self.frame_result:
            ann = {'annotations':[], "class": "image", "filename": im_name}
            img = self.img[im_name] * 255
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for i in range(0, results.shape[0]):
                score = results[i, -2]
                if self.threshold and score < self.threshold:
                    continue
                label = int(results[i, -1])
                if label != 1:
                    continue
                results[i, :][results[i,:] > 1] = 1
                results[i, :][results[i,:] < 0] = 0  
                xmin = int(round(results[i, 0] * img.shape[1]))
                ymin = int(round(results[i, 1] * img.shape[0]))
                xmax = int(round(results[i, 2] * img.shape[1]))
                ymax = int(round(results[i, 3] * img.shape[0]))

                im_crop = img[ymin:ymax, xmin:xmax, :]
                red, seg = self.get_seg(im_crop)
                cv2.imwrite(os.path.join(save_dir,im_name[:-4]+'_'+str(xmin)+'_'+str(xmax)+'_'+str(ymin)+'_'+str(ymax)+'_mask.jpg'), red)
                cv2.imwrite(os.path.join(save_dir,im_name[:-4]+'_'+str(xmin)+'_'+str(xmax)+'_'+str(ymin)+'_'+str(ymax)+'_seg.jpg'), seg)
                coords = [xmin, ymin, xmax - xmin, ymax - ymin]
                ann['annotations'].append({
                        "class": "rect",
                        "height": coords[3],
                        "width": coords[2],
                        "x": coords[0],
                        "y": coords[1]
                        })
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 3)
                display_text = '%s: %.2f' % ('person', score)
                cv2.putText(img, display_text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255,255,255), thickness=3)
            anns.append(ann)
            cv2.imwrite(os.path.join(save_dir, im_name[:-4] + '_dets.jpg'), img)
            print('Saved: ' + os.path.join(save_dir, im_name[:-4] + '_dets.jpg'))
            sys.stdout.flush()
        with open(os.path.join(save_dir, 'multi_person.json'), 'w') as f:
            json.dump(anns, f, indent=2)
    
    def get_output(self, img_dir, save_dir):
        self.detect(img_dir)
        self.save_results(save_dir)


def net_init(batch_size, gpuid=0):
    '''
    @description: load detection & openpose & segementation models
    @param {None} 
    @return: three instances of det_net, op_net, seg_net
    '''
    # load detection model
    modelDeployFile = 'models/detection/res18_deploy.prototxt'
    
    modelWeightsFile = 'models/detection/coco_refinedet_resnet18_1024x1024_iter_163000.caffemodel'
    opt = parser.parse_args()
    opt.model_name = 'seg_shuffle'
    opt.which_model = 'best'
    opt.train_root = 'models/'  # '/public_datasets/.ck'
    opt.resume = True
    opt.typ='conv'
    opt.gpus = gpuid
    seg_model = SegmentModel(opt, False)
    det_net = PeopleDetection(modelDeployFile, modelWeightsFile, seg_model, gpuid=gpuid, img_resize=1024, batch_size=batch_size, threshold=0.60)

    return det_net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "run test and get all result")
    parser.add_argument("--gpuid",
        help = "The gpu chosen to run the model.", required=True)
    parser.add_argument("--save_dir",
        help = "The output directory where we store the result.", required=True)
    parser.add_argument("--test_set", 
        help = "which sets your wanna run test.", required=True)

    args = parser.parse_args()
    # gpu preparation
    assert len(args.gpuid) == 1, "You only need to choose one gpu. But {} gpus are chosen.".format(args.gpuid)
    
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_dir = args.test_set
    if not os.path.exists(img_dir):
        print("{} doesn't exists".format(img_dir))
        sys.exit(0)

    det = net_init(1, gpuid=args.gpuid)
    det.get_output(img_dir, save_dir)
