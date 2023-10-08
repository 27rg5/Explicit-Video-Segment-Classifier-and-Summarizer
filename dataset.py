import os
import pdb
import glob
import torch
import pickle
import numpy as np
from video_utils import EncodeVideo
from torch.utils.data import Dataset, DataLoader


class VideoClipDataset(Dataset):
    def __init__(self, **dataset_dict):
        """
            Description: A unified dataset for all the modalities i.e. video, text and audio

            Params extracted from unpacking dataset_dict
            @param root_dir_path: The directory one level above processed_data which contains raw data as well which hasn't been split
            @param encoded_videos: The folders which have audio and video encodings
                                    Structure: processed_data/encoded_videos
                                                - explicit
                                                    - God Bless America (Video name)
                                                        - video_encs (I have made 1 minute chunks of any that exceeds 95 secs because the RAM gets killed/encodings don't get saved)
                                                            god_bless_america_video_enc_0
                                                            god_bless_america_video_enc_1
                                                                        .
                                                                        .
                                                                        .
                                                            god_bless_america_video_enc_n

                                                        - audio_encs
                                                            god_bless_america_audio_enc_0
                                                            god_bless_america_audio_enc_1
                                                                        .
                                                                        .
                                                                        .
                                                            god_bless_america_audio_enc_n
            @param device: "cuda" or "cpu"
        """
    
        self.root_dir_path, self.encoded_videos, self.EncodeVideo_obj, self.device = dataset_dict.values()
        self.all_non_encoded_videos_path = os.path.join(self.root_dir_path,'processed_data/non_encoded_videos')
        self.classes = {elem.split('/')[-1]:i for i, elem in enumerate(glob.glob(os.path.join(self.root_dir_path,'processed_data/encoded_videos/*')))} #Map class name to id

    def __getitem__(self, index):
        audio_enc_path = glob.glob(os.path.join(self.encoded_videos[index], 'audio_encs/*'))
        assert len(audio_enc_path)==1
        audio_enc_path = audio_enc_path[0]
        class_audio_enc = audio_enc_path.split('/')[-4]
        video_path = os.path.join(self.all_non_encoded_videos_path, class_audio_enc, self.encoded_videos[index].split('/')[-1])
        spectrogram_enc_path = glob.glob(os.path.join(self.encoded_videos[index], 'spectro_encs/*'))
        assert len(spectrogram_enc_path)==1
        spectrogram_enc_path = spectrogram_enc_path[0]
        spectrogram_enc = pickle.load(open(spectrogram_enc_path,'rb'))
        video_enc = self.EncodeVideo_obj.get_video(video_path)
        audio_enc = pickle.load(open(audio_enc_path,'rb'))['processed_speech']

        video_enc = [elem.to(self.device) for elem in video_enc]
        audio_enc = {key:audio_enc[key].to(self.device) for key in audio_enc.keys()}
        #if spectrogram_enc:
        spectrogram_enc = torch.from_numpy(spectrogram_enc).to(self.device)
        class_audio_enc = self.classes[class_audio_enc]
        class_audio_enc = torch.tensor(class_audio_enc).to(self.device)
        return self.encoded_videos[index], video_enc, audio_enc, spectrogram_enc, class_audio_enc
        #return self.encoded_videos[index], video_enc, audio_enc, class_audio_enc
        #return self.encoded_videos[index], video_enc, class_audio_enc
        #return self.encoded_videos[index], audio_enc, class_audio_enc
        #return self.encoded_videos[index], spectrogram_enc, class_audio_enc

    def __len__(self):
        return len(self.encoded_videos)


if __name__=='__main__':
    root_dir_path = os.path.join(os.path.expanduser('~'), 'cls_data')
    dataset_dict = {
        'root_dir_path':root_dir_path,
        'all_videos':glob.glob(os.path.join(os.path.join(root_dir_path, 'processed_data/encoded_videos'))),
        'encodevideo_obj':EncodeVideo(),
        'device':torch.device('cuda:0')
    }
    videoclipdataset = VideoClipDataset(**dataset_dict)
    videoclipdataloader = DataLoader(videoclipdataset, batch_size=1)
    for data in videoclipdataloader:
        video_enc, audio_enc, class_ = data        

