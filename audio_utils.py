import os
import cv2
import pdb
import glob
import torch
import pickle
import librosa
import numpy as np
from tqdm import tqdm
from skimage.util import img_as_float32
from skimage.transform import resize
from skimage.color import gray2rgb
from torchvision.transforms import Resize
import moviepy.editor as mp 
#class GetSpectrogramFromAudio:

def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        return True
    return False

class GetSpectrogramFromAudio:
    def __init__(self):
        self.threshold = 150
        self.temp_save_path = os.path.join(os.path.expanduser('~'), 'temp.wav')
    
    def get_spectrogram(self, vid_file_name):
        #fn = chr(95).join(vid.name.split(chr(46))[:-1])
        #Convert to WAV
        clip = mp.VideoFileClip(str(vid_file_name))
        # if clip.duration > self.threshold:
        #     raise ValueError('The video is greater than 5 minutes')
        
        clip.audio.write_audiofile(self.temp_save_path)

        #Get Spectrogram from wav
        y, sr = librosa.load(self.temp_save_path)
        mel_spec, _ = librosa.effects.trim(y)

        S = librosa.feature.melspectrogram(y=mel_spec)
        log_mel_spect = librosa.power_to_db(S, ref=np.max)
        log_mel_spect_col = resize(log_mel_spect, (128, 1000))
        log_mel_spect_col = (log_mel_spect_col - np.min(log_mel_spect_col))/(np.max(log_mel_spect_col) - np.min(log_mel_spect_col))
        log_mel_spect_col = gray2rgb(log_mel_spect_col)
        log_mel_spect_col = np.transpose(log_mel_spect_col, axes=[2,0,1])
        # log_mel_spect = self.resize(torch.from_numpy(log_mel_spect).unsqueeze(0).unsqueeze(0)).squeeze(0)
        #print('For video file:{} Log mel spect min:{} and max:{} and shape:{}'.format(vid_file_name, torch.min(log_mel_spect), torch.max(log_mel_spect), log_mel_spect.size()))
        return log_mel_spect_col



if __name__ == '__main__':
    root_dir_path = os.path.join(os.path.expanduser('~'), 'cls_data')
    GetSpectrogramFromAudio_obj = GetSpectrogramFromAudio()
    all_videos = glob.glob(os.path.join(root_dir_path,'processed_data/non_encoded_videos/*/*')) 
    encoded_video_path = os.path.join(root_dir_path,'processed_data/encoded_videos')
    encoded_videos_names = [elem.split('/')[-1] for elem in glob.glob(os.path.join(root_dir_path,'processed_data/encoded_videos/*/*')) if len(glob.glob(os.path.join(elem, 'audio_encs/*')))==1]

    videos_filtered = list()
    print('Filtering videos...')
    for video in tqdm(all_videos):
        video_name = video.split('/')[-1]
        if video_name in encoded_videos_names:
            videos_filtered.append(video)

    print('Filtered videos')
    for video in tqdm(videos_filtered): 
        class_ = video.split('/')[-2]
        video_name = video.split('/')[-1]
        audio_enc_path = glob.glob(os.path.join(encoded_video_path, class_, video_name, 'audio_encs/*'))
        assert len(audio_enc_path)==1
        audio_path = os.path.join(encoded_video_path, class_, video_name,'spectro_encs')
        makedir(audio_path)
        ext_ = os.path.splitext(video_name)[1]
        save_path = os.path.join(audio_path, video_name.replace(ext_,'_spectro_enc'))
        if os.path.exists(save_path):
            continue
        spectrogram = GetSpectrogramFromAudio_obj.get_spectrogram(video)
        pickle.dump(spectrogram, open(save_path, 'wb'))