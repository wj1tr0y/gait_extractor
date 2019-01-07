'''
@Author: Jilong Wang
@Date: 2019-01-07 16:08:19
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-07 16:15:19
@Description: file content
'''
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os


# Load a Caffe Model

protoFile = "./models/body_25/pose_deploy.prototxt"
weightsFile = "./models/body_25/pose_iter_584000.caffemodel"


# Specify number of points in the model 
nPoints = 25
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24]]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


# Read Image
im = cv2.imread("./videoframe-011_scene1_bg_H_045_1/frame120.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
ratio = 128.0 / im.shape[0]
inWidth = int(round(im.shape[1] * ratio))
inHeight = int(round(im.shape[0] * ratio))

# Convert image to blob
netInputSize = (inWidth, inHeight)
inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
net.setInput(inpBlob)

# Run Inference (forward pass)
output = net.forward()
# Display probability maps
# plt.figure()
# plt.title('Probability Maps of Keypoints')
# for i in range(nPoints):
#     probMap = output[0, i, :, :]
#     displayMap = cv2.resize(probMap, (inWidth, inHeight), cv2.INTER_LINEAR)
#     plt.subplot(3, 5, i+1); plt.axis('off'); plt.imshow(displayMap, cmap='jet')

# plt.show()
# Extract points

# # X and Y Scale
scaleX = float(im.shape[1]) / output.shape[3]
scaleY = float(im.shape[0]) / output.shape[2]

# # Empty list to store the detected keypoints
points = []

# # Confidence treshold 
threshold = 0.1

for i in range(nPoints):
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

print(len([x for x in points if x is not None]))

# # Display Points & Skeleton

imPoints = im.copy()
imSkeleton = im.copy()
# Draw points
for i, p in enumerate(points):
    cv2.circle(imPoints, p, 8, (255, 255,0), thickness=-1, lineType=cv2.FILLED)
plt.figure(); plt.axis('off'); plt.imshow(imPoints)
plt.show()

# # Draw skeleton
# for pair in POSE_PAIRS:
#     partA = pair[0]
#     partB = pair[1]

#     if points[partA] and points[partB]:
#         cv2.line(imSkeleton, points[partA], points[partB], (255, 255,0), 2)
#         cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

# plt.figure(figsize=(20,10))
# plt.subplot(121); plt.axis('off'); plt.imshow(imPoints)
# #plt.title('Displaying Points')
# plt.subplot(122); plt.axis('off'); plt.imshow(imSkeleton)
# #plt.title('Displaying Skeleton')
# plt.show()
