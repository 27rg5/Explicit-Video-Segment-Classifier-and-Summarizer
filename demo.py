import os
import pdb
import glob
import math
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from summarizer import summarize
from models import VideoModel
from video_utils import EncodeVideo
from moviepy.editor import VideoFileClip
from audio_utils import GetSpectrogramFromAudio
from text_utils import GetTextFromAudio, TokenizeText
from models import LanguageModel, UnifiedModel, SpectrogramModel
from transformers import AutoProcessor, AutoModelForCausalLM
processor = AutoProcessor.from_pretrained("microsoft/git-large-vatex")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vatex")

def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        return True
    return False

def convert_avi_to_mp4(avi_file_path, output_name):
     os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
     return True


def divide_video(video_path, EncodeVideo_obj, UnifiedModel_obj, TokenizeText_obj, GetSpectrogramFromAudio_obj, GetTextFromAudio_obj, classes_reverse_map, device):
    softmax = nn.Softmax(dim=1)
    ext = os.path.splitext(video_path)[1]
    if ext=='.avi':
        output_file = video_path.replace(ext,'')
        convert_avi_to_mp4(video_path, output_file)
        output_file = video_path.replace(ext,'.mp4')
    else:
        output_file = video_path

    video_clip = VideoFileClip(output_file)
    clip_duration = video_clip.duration

    save_dir_path = os.path.join(os.path.expanduser('~'),'chunks_{}'.format(output_file.split('/')[-1]))
    makedir(save_dir_path)

    num_chunks_trail,  num_chunks_lead = math.modf(clip_duration/60)
    num_chunks_lead = int(num_chunks_lead)
    num_chunks_trail = int(num_chunks_trail*60)
    print(clip_duration,' ',num_chunks_lead,' ',num_chunks_trail)

    if num_chunks_lead == 0:
        start_time = 0
        end_time = video_clip.duration
        segment_clip = video_clip.subclip(start_time, end_time)
        path = os.path.join(save_dir_path, output_file.split('/')[-1].replace('.mp4', '_{}.mp4'.format(0)))
        segment_clip.write_videofile(path)
        processed_video = [elem.to(device) for elem in EncodeVideo_obj.get_video(path)]

        with torch.no_grad():
            processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(path))
            processed_speech = {key:processed_speech[key].to(device) for key in processed_speech}
            #processed_spectro = torch.from_numpy(GetSpectrogramFromAudio_obj.get_spectrogram(video_path)).to(device)
            processed_spectro = torch.from_numpy(GetSpectrogramFromAudio_obj.get_spectrogram(path)).to(device)
            predictions = UnifiedModel_obj(processed_speech, processed_video, processed_spectro)
            pred_softmax = softmax(predictions)
            pred_softmax = torch.argmax(pred_softmax, dim=1).cpu().item()
            print('The predicted class is :{}'.format(classes_reverse_map[pred_softmax]))
            if classes_reverse_map[pred_softmax]=='explicit':
                summarized_string = summarize(path, model, processor, device)
                print('The summary is :{}'.format(summarized_string))

        os.remove(path)
        os.system('rm -rf *.mp*')


    else:
        for chunk in tqdm(range(num_chunks_lead)):
            segment_clip_path = os.path.join(save_dir_path, output_file.split('/')[-1].replace('.mp4', '_{}.mp4'.format(chunk)))
            
            start_time = chunk*60
            end_time = start_time + 60
            #pdb.set_trace()
            segment_clip = video_clip.subclip(start_time, end_time)
#            try:
            segment_clip.write_videofile(segment_clip_path)
            # except:
            #     pdb.set_trace()
            #segment_clip.close()
            processed_video = [elem.to(device) for elem in EncodeVideo_obj.get_video(segment_clip_path)]
            with torch.no_grad():
                #processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(video_path))
                processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(segment_clip_path))
                processed_speech = {key:processed_speech[key].to(device) for key in processed_speech}
                #processed_spectro = torch.from_numpy(GetSpectrogramFromAudio_obj.get_spectrogram(video_path)).to(device)
                processed_spectro = torch.from_numpy(GetSpectrogramFromAudio_obj.get_spectrogram(segment_clip_path)).to(device)
                predictions = UnifiedModel_obj(processed_speech, processed_video, processed_spectro)
                pred_softmax = softmax(predictions)
                pred_softmax = torch.argmax(pred_softmax, dim=1).cpu().item()
                print('The predicted class is :{} for time segment:{} to {}'.format(classes_reverse_map[pred_softmax], start_time, end_time))
                if classes_reverse_map[pred_softmax]=='explicit':
                    summarized_string = summarize(segment_clip_path, model, processor, device)
                    print('The summary is :{}'.format(summarized_string))

            os.remove(segment_clip_path)
            os.system('rm -rf *.mp*')

        #for chunk in tqdm(range(num_chunks_trail)):
        if num_chunks_trail!=0:
            start_time = end_time
            end_time = start_time + num_chunks_trail
            segment_clip_path = os.path.join(save_dir_path, output_file.split('/')[-1].replace('.mp4', '_{}.mp4'.format(chunk)))
            
            segment_clip = video_clip.subclip(start_time, end_time)
            segment_clip.write_videofile(segment_clip_path)
            #segment_clip.close()
            processed_video = [elem.to(device) for elem in EncodeVideo_obj.get_video(segment_clip_path)]
            
            with torch.no_grad():
                #processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(video_path))
                processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(segment_clip_path))
                processed_speech = {key:processed_speech[key].to(device) for key in processed_speech}
                #processed_spectro = torch.from_numpy(GetSpectrogramFromAudio_obj.get_spectrogram(video_path)).to(device)
                processed_spectro = torch.from_numpy(GetSpectrogramFromAudio_obj.get_spectrogram(segment_clip_path)).to(device)
                predictions = UnifiedModel_obj(processed_speech, processed_video, processed_spectro)
                pred_softmax = softmax(predictions)
                pred_softmax = torch.argmax(pred_softmax, dim=1).cpu().item()
                print('The predicted class is :{} for time segment:{} to {}'.format(classes_reverse_map[pred_softmax], start_time, end_time))
                if classes_reverse_map[pred_softmax]=='explicit':
                    summarized_string = summarize(segment_clip_path, model, processor, device)
                    print('The summary is :{}'.format(summarized_string))

            os.remove(segment_clip_path)
            os.system('rm -rf *.mp*')


    shutil.rmtree(save_dir_path)

def inference(stitched_videos_path, classes_reverse_map, checkpoint_path, experiment_name=None, language_model_name='distilbert-base-uncased', video_model_name='slowfast_r50', spectrogram_model_name='resnet18', device='cuda:0'):
    LanguageModel_obj = LanguageModel(model_name = language_model_name, demo=True)
    VideoModel_obj = VideoModel(model_name = video_model_name,  demo=True)
    SpectrogramModel_obj = SpectrogramModel(model_name = spectrogram_model_name,  demo=True)
    batch_size = 1
    softmax = nn.Softmax(dim=1)
    in_dims = 600
    intermediate_dims = 50
    UnifiedModel_obj = UnifiedModel(in_dims, intermediate_dims, LanguageModel_obj, VideoModel_obj, SpectrogramModel_obj).to(device)
    
    UnifiedModel_obj.load_state_dict(torch.load(checkpoint_path), strict=True)
    UnifiedModel_obj.eval()

    EncodeVideo_obj = EncodeVideo()
    GetTextFromAudio_obj = GetTextFromAudio()
    GetSpectrogramFromAudio_obj = GetSpectrogramFromAudio()
    TokenizeText_obj = TokenizeText()    

    videos = glob.glob(os.path.join(stitched_videos_path, 'Lord.of.War/*'))
    for video_path in videos:
        print('For video :{}'.format(video_path))
        divide_video(video_path, EncodeVideo_obj, UnifiedModel_obj, TokenizeText_obj, GetSpectrogramFromAudio_obj, GetTextFromAudio_obj, classes_reverse_map, device)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stitched_videos_path', type=str, default=os.path.join(os.path.expanduser('~'), 'cls_data'))
    parser.add_argument('--experiment_name', type=str, default='third_run_sgd_lr_1e-3_macro_f1_with_seed_42')
    parser.add_argument('--get_classified_list', action='store_true')

    
    args = parser.parse_args()
    stitched_videos_path = args.stitched_videos_path
    experiment_name = args.experiment_name
    classes = {'explicit': 0, 'non_explicit': 1}
    classes_reverse_map = {v:k for k,v in classes.items()}
    checkpoint_path = os.path.join(os.getcwd(),'runs',experiment_name, 'best_checkpoint.pth')

    inference(stitched_videos_path, classes_reverse_map, checkpoint_path)


