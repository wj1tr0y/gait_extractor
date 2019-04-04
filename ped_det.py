'''
@Author: Jilong Wang
@Date: 2019-01-05 17:47:31
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-03-11 18:09:13
@Description: pedestrain detection
'''
import cv2
import argparse
import os
import shutil
import subprocess
import zipfile
import time
import sys
import detector

def split_video(video_name, frame_save_dir):
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if not os.path.exists(frame_save_dir):
        os.makedirs(frame_save_dir)
        frame_count = 1
        success = True
        while(success):
            success, frame = cap.read()
            if success:
                print 'Reading frames: {}\r'.format(frame_count),
                cv2.imwrite(os.path.join(frame_save_dir, 'frame{}.jpg'.format(frame_count)), frame)
                frame_count += 1
            else:
                print ''
        cap.release()
    else:
        print('Video had already split in frames stored in {}'.format(frame_save_dir))
    return fps, size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "run test and get all result")
    parser.add_argument("video", 
        help = "path of test video")
    parser.add_argument("--gpuid",
        help = "The gpu chosen to run the model.", required=True)

    args = parser.parse_args()
    assert len(args.gpuid) == 1, "You only need to choose one gpu. But {} gpus are chosen.".format(args.gpuid)

    
    SET_TEST = False
    if os.path.isdir(args.video):
        SET_TEST = True
    elif not os.path.exists(args.video):
        print "{} doesn't exist.".format(args.video)

    video_names = [args.video]
    if SET_TEST:
        video_names = os.listdir(args.video)
        video_names = [os.path.join(args.video, x) for x in video_names]
    
    modelDeployFile = 'models/detection/res18_deploy.prototxt'
    modelWeightsFile = 'models/detection/coco_refinedet_resnet18_addneg_1024x1024_iter_340000.caffemodel'
    
    # initialize detect net 
    pedes_det = detector.PeopleDetection(modelDeployFile, modelWeightsFile, gpuid=args.gpuid, img_resize=1024, batch_size=1, threshold=0.40)

    time_cost = 0
    for video_name in video_names:
        frame_save_dir = './videoframes/videoframe-'+ os.path.basename(video_name)[:-4]

        # split video into frame pictures
        fps, size = split_video(video_name, frame_save_dir)

        img_dir = frame_save_dir
        if not os.path.exists(img_dir):
            print("{} doesn't exists".format(img_dir))
            sys.exit(0)

        # result save path
        basename = os.path.basename(video_name)[:-4]
        save_dir = os.path.join('./results/',basename)

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        start_time = time.time()
        print 'Detecting.....'
        
        # detection
        pedes_det.get_output(img_dir, save_dir)

        # render pics into video
        print('Detection done. Now render results into video file.')
        out_video_name = basename +'_dets'+ '.avi'

        videoWriter = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc('X','V','I','D'), int(fps), size)
        frame_name = os.listdir(save_dir)
        frame_name = sorted(frame_name, key=lambda x: int(x[5:-9]))
        for i in frame_name:
            frame = cv2.imread(os.path.join(save_dir, i))
            videoWriter.write(frame)
        videoWriter.release()
        
        # shutil.rmtree(save_dir)

        # cmd = "ffmpeg -threads 12 -y -i {} -strict experimental {}".format(out_video_name, out_video_name[:-4]+'.mp4')
        # print(cmd)
        # process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        # output = process.communicate()
        # print(output)
        print('Done. Result was stored in {}'.format(out_video_name))
        
        time_cost += time.time()-start_time
    print(time_cost)
