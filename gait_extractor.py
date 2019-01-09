'''
@Author: Jilong Wang
@Date: 2019-01-05 17:47:31
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-09 11:25:36
@Description: Gait extractor. Supporting single video file extraction{pass the video file path} and mutli-videos extraction{pass the video folder path}
'''
import cv2
import argparse
import os
import shutil
import subprocess
import zipfile
import time
import sys

def split_video(video_name, frame_save_dir):
    if not os.path.exists(frame_save_dir):
        os.mkdir(frame_save_dir)
        cap = cv2.VideoCapture(video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "run test and get all result")
    parser.add_argument("video", 
        help = "path of test video")
    parser.add_argument("dataset",
        help = "use which dataset.")
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
    
    if args.dataset == 'casia_b':
        import casia_b as casia
    elif args.dataset == 'casia_e':
        import casia_e as casia

    # initialize openpose and detect net 
    gait_extractor = casia.GaitExtractor(args.gpuid, det_batch_size=20)
    cost = 0
    for video_name in video_names:
        frame_save_dir = './videoframes/videoframe-'+ os.path.basename(video_name)[:-4]

        # split video into frame pictures
        split_video(video_name, frame_save_dir)

        img_dir = frame_save_dir
        if not os.path.exists(img_dir):
            print("{} doesn't exists".format(img_dir))
            sys.exit(0)

        # gait picture save path
        save_dir = './results/result-' + os.path.basename(video_name)[:-4]
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)

        start_time = time.time()
        print 'Extracting gait.....'
        
        # do extraction
        gait_extractor.extract(img_dir, save_dir)
        
        cost += time.time()-start_time
    print(cost)