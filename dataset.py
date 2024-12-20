import os
import pdb
import random
import glob
import torch
import pickle
import numpy as np
from tqdm import tqdm
from video_utils import EncodeVideo
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):

    video_paths, video_encs, audio_encs, spectrogram_encs, captions, classes = zip(*batch)
    
    #Breakpoint
    #pdb.set_trace()
    if isinstance(video_encs[0], list):
        fast_path = torch.stack([video[0].squeeze(0) for video in video_encs])
        slow_path = torch.stack([video[1].squeeze(0) for video in video_encs])
        video_encs = [fast_path, slow_path]
    else:
        video_encs = torch.Tensor(video_encs)
    
    if isinstance(spectrogram_encs[0], int):
        spectrogram_encs = torch.Tensor(spectrogram_encs) 
    else:
        spectrogram_encs = torch.stack(spectrogram_encs)

    if not isinstance (audio_encs[0], int):
        input_ids = [enc['input_ids'].squeeze(0) for enc in audio_encs]
        attention_masks = [enc['attention_mask'].squeeze(0) for enc in audio_encs]

        # Pad sequences to the same length
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        # Create a new BatchEncoding with the padded sequences
        audio_encs_padded = {
            'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded
        }
    else:
        audio_encs_padded = torch.Tensor(audio_encs)
    # Batch captions (if needed, depeinput_idsnding on their format)
    if not isinstance(captions[0], int):
        captions = torch.stack(captions)  # Assumes captions are tensors of uniform size
    else:
        captions = torch.Tensor(captions)
    # Batch targets (classes)
    
    classes = torch.tensor(classes)  # Shape: [batch_size]

    return video_paths, video_encs, audio_encs_padded, spectrogram_encs, captions, classes

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
    
        self.root_dir_path, self.encoded_videos, self.EncodeVideo_obj, self.device, self.modalities, self.caption_df_dict = dataset_dict.values()
        self.classes = {elem.split('/')[-1]:i for i, elem in enumerate(sorted(glob.glob(os.path.join(self.root_dir_path,'encoded_videos/*'))))} #Map class name to id

        # Create self.labels corresponding to each video in self.encoded_videos
        self.labels = []
        for video_path in self.encoded_videos:
            class_str = video_path.split('/')[-4]  # Extract class name from the path
            class_ = self.classes[class_str]       # Get class ID
            self.labels.append(class_)             # Append class ID to labels list        
    
    def __getitem__(self, index):
        video_path = self.encoded_videos[index]
        class_ = self.labels[index]
        
        # Fetch encodings
        video_enc, audio_enc, spectrogram_enc, caption = 0, 0, 0, 0
        subclip_num, ext = video_path.split('/')[-1].split('_')[-1].split('.')

        if self.caption_df_dict:
            try:
                caption = torch.Tensor(self.caption_df_dict[video_path][1])
            except KeyError:
                pass

        if 'video' in self.modalities:
            video_enc = self.EncodeVideo_obj.get_video(video_path)

        if 'text' in self.modalities:
            audio_enc_path = video_path.replace('video_subclips', 'audio_encs').replace(f'_{subclip_num}.{ext}', f'_audio_enc_{subclip_num}')
            audio_enc = pickle.load(open(audio_enc_path, 'rb'))['processed_speech']

        if 'audio' in self.modalities:
            spectrogram_enc_path = video_path.replace('video_subclips', 'spectro_encs').replace(f'_{subclip_num}.{ext}', f'_spectro_enc_{subclip_num}')
            spectrogram_enc = torch.from_numpy(pickle.load(open(spectrogram_enc_path, 'rb'))['processed_spectro'])

        #Breakpoint
        #pdb.set_trace()
        return video_path, video_enc, audio_enc, spectrogram_enc, caption, class_

    def __len__(self):
        return len(self.encoded_videos)

    
if __name__=='__main__':
    exp_dir_with_captions = 'runs/attention_fusion_default_networks_self_attention_21epochs_caption_modality'
    root_dir_path = os.path.join(os.path.expanduser('~'), 'cls_data_1_min')
    encoded_videos = pickle.load(open(os.path.join(exp_dir_with_captions,'val_encoded_video.pkl'),'rb'))
    
    device = torch.device('cpu')
    modalities = ['text','video','audio']
    all_captions_dict = pickle.load(open(os.path.join(exp_dir_with_captions,'all_captions.pkl'),'rb'))
    EncodeVideo_obj = EncodeVideo()
    dataset_dict = {
        'root_dir':root_dir_path,
        'all_encoded_videos':encoded_videos,
        'encoded_video_obj':EncodeVideo_obj,
        'device':device,
        'modalities':modalities,
        'all_captions_dict':all_captions_dict
    }
    videoclipdataset = VideoClipDataset(**dataset_dict)

    labels_np = np.array(videoclipdataset.labels)
    class_counts = np.bincount(labels_np)
    class_weights = 1.0 / class_counts  # Inverse frequency

    # Create a mapping from class ID to class weight
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    # Assign a weight to each sample in the dataset
    sample_weights = [class_weight_dict[label] for label in videoclipdataset.labels]
    
    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # You can adjust this number if needed
        replacement=True  # Set to True to allow sampling with replacement
    )    
    videoclipdataloader = DataLoader(videoclipdataset, batch_size=5, sampler=sampler, collate_fn=collate_fn)
    for i, modality_inputs in tqdm(enumerate(videoclipdataloader)):
        video_path, transformed_video, processed_speech, spectrogram, caption, target = modality_inputs
        if not (transformed_video[0].ndim==5 and transformed_video[1].ndim==5 and processed_speech['input_ids'].ndim==2 and processed_speech['attention_mask'].ndim==2 and spectrogram.ndim==4):
            pdb.set_trace()
            

