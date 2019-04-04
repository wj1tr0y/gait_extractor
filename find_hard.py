#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jilong Wang
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@Description: file content
@Date: 2019-03-14 12:05:27
@LastEditTime: 2019-03-14 12:10:03
'''
import threading
import json
import os
import threading
import shutil
from tqdm import tqdm

def compute_iou(rec1, rec2):
    one_x, one_y, one_w, one_h = rec1
    two_x, two_y, two_w, two_h = rec2
    if((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
        lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
        lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))

        rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
        rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))

        inter_w = abs(rd_x_inter - lu_x_inter)
        inter_h = abs(lu_y_inter - rd_y_inter)

        inter_square = inter_w * inter_h
        union_square = (one_w * one_h) + (two_w * two_h) - inter_square

        calcIOU = inter_square / union_square * 1.0
        return calcIOU
    else:
        return 0

def find_hard():
    hard_name = []
    try:
        ann = {
  "annotation": [
    {
      "bbox": [
        276,
        269,
        34,
        67
      ],
      "category_id": 1
    },
    {
      "bbox": [
        24,
        66,
        34,
        93
      ],
      "category_id": 1
    },
    {
      "bbox": [
        195,
        376,
        36,
        88
      ],
      "category_id": 1
    }
  ],
  "image": {
    "file_name": "set05_V007_I00999.jpg",
    "height": 480,
    "width": 640
  }
}
        result = [
  {
    "bbox": [
      194,
      373,
      228,
      464
    ]
  },
  {
    "bbox": [
      21,
      67,
      58,
      158
    ]
  },
  {
    "bbox": [
      286,
      276,
      310,
      336
    ]
  }
]
        ann = ann['annotation']

        for res in result:
            res['count'] = 0
        for gt in ann:
            gt['count'] = 0

        for i in range(len(result)):
            bbox2 = result[i]['bbox']
            rect2 = [bbox2[0], bbox2[1], bbox2[2]-bbox2[0], bbox2[3]-bbox2[1]]
            max_iou = (-1, 0)
            for j in range(len(ann)):
                bbox = ann[j]['bbox']
                rect = [bbox[0], bbox[1], bbox[2], bbox[3]]
                print(rect, rect2)
                iou = compute_iou(rect, rect2)
                print(iou)
                if iou > max_iou[1] and iou > 0.5:
                    max_iou = (j, iou)
            if max_iou[0] != -1:
                ann[max_iou[0]]['count'] += 1
                result[i]['count'] += 1


        multi_bbox = 0.
        mismatch_bbox = 0.
        lost_bbox = 0.
        for gt in ann:
            if gt['count'] > 1:
                multi_bbox += 1
            if gt['count'] == 0:
                lost_bbox +=1
        for res in result:
            if res['count'] == 0:
                mismatch_bbox += 1
        if len(ann) == 0:
            if mismatch_bbox > 3:
                hard_name.append(det)
        elif mismatch_bbox/len(ann) > 0.5 or multi_bbox/len(ann) > 0.5 or lost_bbox/len(ann) > 0.5:
            print(mismatch_bbox, multi_bbox, lost_bbox)
    except:
        pass

find_hard()