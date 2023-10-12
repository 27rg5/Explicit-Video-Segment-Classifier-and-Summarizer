import os
import glob
import pdb
import torch
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from models import VideoModel
from video_utils import EncodeVideo
from dataset import VideoClipDataset
from torch.utils.data import DataLoader
from models import LanguageModel, UnifiedModel, SpectrogramModel
from torcheval.metrics.functional import multiclass_f1_score

def inference_on_val(non_encoded_videos_path, val_encoded_videos_pkl, classes, checkpoint_path, root_dir_path, EncodeVideo_obj, experiment_name=None, get_classified_list=None, language_model_name='distilbert-base-uncased', video_model_name='slowfast_r50', spectrogram_model_name='resnet18', device='cuda:0'):
    LanguageModel_obj = LanguageModel(model_name = language_model_name)
    VideoModel_obj = VideoModel(model_name = video_model_name)
    SpectrogramModel_obj = SpectrogramModel(model_name = spectrogram_model_name)
    batch_size = 1
    softmax = nn.Softmax(dim=1)
    in_dims = 600
    #in_dims = 500
    intermediate_dims = 50
    UnifiedModel_obj = UnifiedModel(in_dims, intermediate_dims, LanguageModel_obj, VideoModel_obj, SpectrogramModel_obj).to(device)
    
    UnifiedModel_obj.load_state_dict(torch.load(checkpoint_path), strict=True)
    UnifiedModel_obj.eval()

    val_videos = pickle.load(open(val_encoded_videos_pkl,'rb'))

    
    val_dataset_dict = {
    'root_dir':root_dir_path,
    'all_encoded_videos':val_videos,
    'encoded_video_obj':EncodeVideo_obj,
    'device':device
    }
    val_dataset = VideoClipDataset(**val_dataset_dict)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    classes = val_dataset.classes
    reverse_class_map = {v:k for k,v in classes.items()}
        

    preds_val = list()
    targets_val = list()
    
    videos = list()
    for i, modality_inputs in tqdm(enumerate(val_dataloader)):
        with torch.no_grad():
            video_path, transformed_video, processed_speech, spectrogram_enc, target = modality_inputs
            target = target.to(device)
            predictions = UnifiedModel_obj(processed_speech, transformed_video, spectrogram_enc)
            pred_softmax = softmax(predictions)
            pred_softmax = torch.argmax(pred_softmax, dim=1)
            preds_val.append(pred_softmax.cpu().item())
            targets_val.append(target.cpu().item())
            videos.append(video_path)
    targets_val = torch.tensor(targets_val)
    preds_val = torch.tensor(preds_val)

    predictions = list()
    if get_classified_list:
        predictions = list()
        for video, pred, target in zip(videos, preds_val, targets_val):
            predictions.append((video, reverse_class_map[pred.item()], reverse_class_map[target.item()]))
        
        predictions_df = pd.DataFrame(predictions, columns=['Video path','Prediction','Target'])
        predictions_df.to_csv(os.path.join(os.getcwd(),'runs',experiment_name,'predictions.csv'), index=False)


    print('f1-score macro:{} f1-score micro:{} f1-score weighted:{} accuracy:{}'.format(multiclass_f1_score(preds_val, targets_val, num_classes=2, average="macro").item(), multiclass_f1_score(preds_val, targets_val, num_classes=2, average="micro").item(), multiclass_f1_score(preds_val, targets_val, num_classes=2, average="weighted").item(), (preds_val==targets_val).sum()/len(targets_val)))
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir_path', type=str, default=os.path.join(os.path.expanduser('~'), 'cls_data'))
    parser.add_argument('--experiment_name', type=str, default='third_run_sgd_lr_1e-3_macro_f1_with_seed_42')
    parser.add_argument('--get_classified_list', action='store_true')
    
    args = parser.parse_args()
    root_dir_path = args.root_dir_path
    experiment_name = args.experiment_name
    get_classified_list = args.get_classified_list
    non_encoded_videos_path = os.path.join(root_dir_path,'processed_data/non_encoded_videos')
    val_encoded_videos_pkl = 'runs/{}/val_encoded_video.pkl'.format(experiment_name)
    classes = {elem.split('/')[-1]:i for i, elem in enumerate(glob.glob(os.path.join(root_dir_path,'processed_data/encoded_videos/*')))}
    checkpoint_path = os.path.join(os.getcwd(),'runs',experiment_name, 'best_checkpoint.pth')
    EncodeVideo_obj = EncodeVideo()
    if get_classified_list:
        inference_on_val(non_encoded_videos_path, val_encoded_videos_pkl, classes, checkpoint_path, root_dir_path, EncodeVideo_obj, get_classified_list=get_classified_list, experiment_name=experiment_name, device='cuda:1')
    else:
        inference_on_val(non_encoded_videos_path, val_encoded_videos_pkl, classes, checkpoint_path, root_dir_path, EncodeVideo_obj, experiment_name=experiment_name)
