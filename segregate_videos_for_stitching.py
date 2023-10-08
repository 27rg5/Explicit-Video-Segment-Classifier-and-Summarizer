import os
import pdb
import glob
import shutil
import pickle
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip

def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)



if __name__=='__main__':
    #pdb.set_trace()
    val_encoded = pickle.load(open(os.path.join(os.getcwd(), 'val_encoded_video.pkl'), 'rb'))
    video_names = [elem.split('/')[-1].split('__')[0] for elem in val_encoded]
    video_source = glob.glob(os.path.join(os.path.expanduser('~'), 'non_encoded_videos/*/*'))
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

