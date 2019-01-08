'''
@Author: Jilong Wang
@Date: 2019-01-05 17:47:31
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-08 18:11:03
@Description: Gait extractor. Supporting single video file extraction{pass the video file path} and mutli-videos extraction{pass the video folder path}
'''
import cv2
import argparse
import os
import shutil
import subprocess
import zipfile
import time

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
    
    cost = 0
    for video_name in video_names:
        frame_save_dir = './videoframes/videoframe-'+ os.path.basename(video_name)[:-4]

        # split video into frame pictures
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

        
        
        # gait picture save path
        out_dir = './results/result-' + os.path.basename(video_name)[:-4]
        
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        start_time = time.time()
        print 'Detecting pedestrian.....'
        cmd = "python {} --gpuid {} --out-dir {} --test-set {}".format(args.dataset+'.py', args.gpuid, out_dir, frame_save_dir)
        print(cmd)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        # output = process.communicate()

        while process.poll() is None:
            line = process.stdout.readline()
            line = line.strip()
            if line:
                print('Subprogram output: [{}]'.format(line))
        cost += time.time()-start_time
    print(cost)