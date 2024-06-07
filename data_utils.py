import warnings
warnings.filterwarnings("ignore")
import cv2
import os
import pdb
import math
import glob
import pickle
from tqdm import tqdm
from video_utils import EncodeVideo
from moviepy.editor import VideoFileClip
from text_utils import GetTextFromAudio, TokenizeText
from audio_utils import GetSpectrogramFromAudio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        return True
    return False

def convert_avi_to_mp4(avi_file_path, output_name):
     os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
     return True

def divide_video_and_save_all_modalities(video_path, save_path_enc_dir, EncodeVideo_obj, TokenizeText_obj, GetSpectrogramFromAudio_obj, GetTextFromAudio_obj):
    
    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps
        video.release()
    except:
        print('\n CORRUPT VIDEO')
        return #Corrupt video
    makedir(save_path_enc_dir)

    save_dir_video = os.path.join(save_path_enc_dir,'video_subclips')
    save_dir_audio = os.path.join(save_path_enc_dir,'audio_encs')
    save_dir_spectro = os.path.join(save_path_enc_dir,'spectro_encs')
    makedir(save_dir_spectro)
    makedir(save_dir_video)
    makedir(save_dir_audio)

    ext = os.path.splitext(video_path)[1]
    if ext=='.avi':
        output_file = video_path.replace(ext,'')
        convert_avi_to_mp4(video_path, output_file)
        output_file = video_path.replace(ext,'.mp4')
    else:
        output_file = video_path

    video_clip = VideoFileClip(output_file)
    clip_duration = duration_sec
    num_chunks_trail,  num_chunks_lead = math.modf(clip_duration/60)
    #pdb.set_trace()
    num_chunks_lead = int(num_chunks_lead)
    num_chunks_trail = int(num_chunks_trail*60)
    if num_chunks_lead == 0:
        start_time = 0
        end_time = clip_duration
        segment_clip = video_clip.subclip(start_time, end_time)
        segment_clip_path = os.path.join(save_dir_video, output_file.split('/')[-1].replace('.mp4', '_{}.mp4'.format(0)))
        segment_clip.write_videofile(segment_clip_path)

        processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(segment_clip_path))
        pickle.dump({'processed_speech':processed_speech}, open(os.path.join(save_dir_audio, output_file.split('/')[-1].replace('.mp4', '_audio_enc_{}'.format(0))), 'wb'))
        processed_spectro = GetSpectrogramFromAudio_obj.get_spectrogram(segment_clip_path)
        pickle.dump({'processed_spectro':processed_spectro}, open(os.path.join(save_dir_spectro, output_file.split('/')[-1].replace('.mp4', '_spectro_enc_{}'.format(0))), 'wb'))

    
    else:
        for chunk in range(num_chunks_lead):
            segment_clip_path = os.path.join(save_dir_video, output_file.split('/')[-1].replace('.mp4', '_{}.mp4'.format(chunk)))
            
            start_time = chunk*60
            end_time = start_time + 60
            #pdb.set_trace()
            segment_clip = video_clip.subclip(start_time, end_time)
#            try:
            segment_clip.write_videofile(segment_clip_path)
            processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(segment_clip_path))
            pickle.dump({'processed_speech':processed_speech}, open(os.path.join(save_dir_audio, output_file.split('/')[-1].replace('.mp4', '_audio_enc_{}'.format(chunk))), 'wb'))
            processed_spectro = GetSpectrogramFromAudio_obj.get_spectrogram(segment_clip_path)
            pickle.dump({'processed_spectro':processed_spectro}, open(os.path.join(save_dir_spectro, output_file.split('/')[-1].replace('.mp4', '_spectro_enc_{}'.format(chunk))), 'wb'))

    if num_chunks_trail!=0 and num_chunks_lead!=0:
        start_time = end_time
        end_time = start_time + num_chunks_trail
        segment_clip_path = os.path.join(save_dir_video, output_file.split('/')[-1].replace('.mp4', '_{}.mp4'.format(chunk+1)))
        
        segment_clip = video_clip.subclip(start_time, end_time)
        segment_clip.write_videofile(segment_clip_path)
        processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(segment_clip_path))
        pickle.dump({'processed_speech':processed_speech}, open(os.path.join(save_dir_audio, output_file.split('/')[-1].replace('.mp4', '_audio_enc_{}'.format(chunk))), 'wb'))
        processed_spectro = GetSpectrogramFromAudio_obj.get_spectrogram(segment_clip_path)
        pickle.dump({'processed_spectro':processed_spectro}, open(os.path.join(save_dir_spectro, output_file.split('/')[-1].replace('.mp4', '_spectro_enc_{}'.format(chunk))), 'wb'))





def encode_videos(videos_path, encoded_videos_path, EncodeVideo_obj, GetTextFromAudio_obj, GetSpectrogramFromAudio_obj, TokenizeText_obj):
    explicit_encoded = os.path.join(encoded_videos_path,'explicit')
    non_explicit_encoded = os.path.join(encoded_videos_path,'non_explicit')
    makedir(explicit_encoded)
    makedir(non_explicit_encoded)
    print('Encoding all videos and text...')
    skipped_videos = 0
    for video_path in tqdm(videos_path):
        class_ = video_path.split('/')[-2]
        if class_=='explicit':
            use_var = explicit_encoded
        else:
            use_var = non_explicit_encoded
        save_path_enc_dir = os.path.join(use_var, video_path.split('/')[-1])

        divide_video_and_save_all_modalities(video_path, save_path_enc_dir, EncodeVideo_obj, TokenizeText_obj, GetSpectrogramFromAudio_obj, GetTextFromAudio_obj)
        


if __name__=='__main__':
    root_dir = os.path.join(os.path.expanduser('~'), 'cls_data_1_min')
    all_videos = glob.glob(os.path.join(root_dir,'non_encoded_videos_sep/*/*'))
    encoded_videos_path = os.path.join(root_dir,'encoded_videos/')
    EncodeVideo_obj = EncodeVideo()
    GetTextFromAudio_obj = GetTextFromAudio()
    GetSpectrogramFromAudio_obj = GetSpectrogramFromAudio()
    TokenizeText_obj = TokenizeText()
    encode_videos(all_videos, encoded_videos_path, EncodeVideo_obj, GetTextFromAudio_obj, GetSpectrogramFromAudio_obj, TokenizeText_obj)    
