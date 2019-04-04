#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jilong Wang
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@Description: file content
@Date: 2019-03-19 16:50:35
@LastEditTime: 2019-03-28 14:40:04
'''
import cv2
import os
import sys
from tqdm import tqdm
import shutil
import json

class DrawAnnotations:
    def __init__(self, img_dir, ann_dir, save_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.save_dir = save_dir
        self.ann_names = os.listdir(ann_dir)
        # self.img_names = self.img_names[:10]
    def draw(self):
        pbar = tqdm(total=len(self.ann_names))
        for ann_name in self.ann_names:
            ann = json.load(open(os.path.join(self.ann_dir, ann_name), 'r'))
            img = cv2.imread(os.path.join(self.img_dir, ann_name[:-5]+'.jpg'))
            ann = ann['annotation']
            for bbox in ann:
                bbox = bbox['bbox']
                xmin = int(round(bbox[0]))
                ymin = int(round(bbox[1]))
                xmax = int(round(bbox[0]+bbox[2]))
                ymax = int(round(bbox[1]+bbox[3]))
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 3)

            cv2.imwrite(os.path.join(self.save_dir, ann_name[:-5] + '_dets.jpg'), img)
            # pbar.set_description('Saved: ' + os.path.join(self.save_dir, img_name[:-4] + '_dets.jpg'))
            pbar.update()


if __name__ == '__main__':
    img_dir = 'data/ImageSet'
    ann_dir = 'data/Annotations'
    save_dir ='data/Draw_Rectangle'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    drawer = DrawAnnotations(img_dir, ann_dir, save_dir)
    drawer.draw()