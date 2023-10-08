import os
import pdb
import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips

def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


if __name__=='__main__':
    video_dirs = glob.glob(os.path.join(os.getcwd(),'wanted_videos/*'))
    stitched_video_dir = os.path.join(os.getcwd(), 'stitched_videos')
    makedir(stitched_video_dir)

    for video_dir in tqdm(video_dirs):
        video_clips = sorted(glob.glob(os.path.join(video_dir, '*')))
        video_list = list()
        video_dir_stitched_new = video_dir.split('/')[-1]
        video_dir_stitched_new = os.path.join(stitched_video_dir, video_dir_stitched_new)
        makedir(video_dir_stitched_new)
        clips = list()
        for video in video_clips:
            clip = VideoFileClip(video)
            clips.append(clip)
            video_list.append(clip)
        
        final_clip = concatenate_videoclips(video_list)

#        del video_list
        final_clip.write_videofile(os.path.join(video_dir_stitched_new, video_dir.split('/')[-1]+'.mp4'))
        for clip_ in clips:
            clip_.close()

        final_clip.close()