'''
@Author: Jilong Wang
@Date: 2019-01-08 16:43:42
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-09 15:49:58
@Description: file content
'''
import os
import shutil

DATA_ROOT = '/home/wangjilong/data/casia_b'
video_names = os.listdir(os.path.join(DATA_ROOT, 'videos'))
video_names.sort()
names = []
for i in range(0, len(video_names), 38400):
    if i + 38400 > len(video_names):
        names.append(video_names[i:])
    else:
        names.append(video_names[i:i+38400])

for i in range(len(names)):
    os.mkdir(os.path.join(DATA_ROOT, 'slice'+str(i)))
    print("Processing set {}".format(i))
    for v_name in names[i]:
        shutil.copy(os.path.join(DATA_ROOT, 'videos', v_name), os.path.join(DATA_ROOT, 'slice'+str(i), v_name))

