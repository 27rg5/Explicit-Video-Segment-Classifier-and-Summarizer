import os
import pdb
import glob
import shutil
import pickle
import argparse
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip

def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)



if __name__=='__main__':
    #pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='experiment_1')
    args = parser.parse_args()
    experiment_name = args.experiment_name
    val_encoded = pickle.load(open(os.path.join(os.getcwd(), 'runs', experiment_name, 'val_encoded_video.pkl'), 'rb'))
    video_names = [elem.split('/')[-1].split('__')[0] for elem in val_encoded]
    #video_source = glob.glob(os.path.join(os.path.expanduser('~'), 'non_encoded_videos/*/*'))
    video_source = glob.glob(os.path.join(os.path.expanduser('~'), 'cls_data_1_min/encoded_videos/*/*/video_subclips/*'))
    save_path = os.path.join(os.getcwd(), 'wanted_videos')
    makedir(save_path)

    for video in tqdm(video_source):
        ext_ = os.path.splitext(video)[1]
        if ext_=='.avi':
            continue
        
        curr_vid_name = video.split('/')[-1].split('__')[0]
        if curr_vid_name in video_names:
            clip = VideoFileClip(video)
            if clip.duration > 60:
                clip.close()
                continue
            save_vid_path = os.path.join(save_path, curr_vid_name)
            makedir(save_vid_path)
            shutil.copy(video, save_vid_path)
            clip.close()

