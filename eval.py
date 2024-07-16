import os
import glob
import pdb
import csv
import torch
import random
import joblib
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from models import VideoModel
from video_utils import EncodeVideo
from dataset import VideoClipDataset
from summarizer import summarize
from torch.utils.data import DataLoader
#from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCausalLM
from models import LanguageModel, UnifiedModel, SpectrogramModel
from torcheval.metrics.functional import multiclass_f1_score

#Set all seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

#Set deterministic pytorch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

#For cuda version>=10.2
if float(torch.version.cuda) >= 10.2:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8" #or ":16:8"

def inference_on_val(videos_pkl, eval_dataset_type, classes, checkpoint_path, root_dir_path, EncodeVideo_obj, modalities=['video','audio','text'], vanilla_fusion=False, pairwise_attention_modalities=False, experiment_name=None, get_classified_list=None, language_model_name='distilbert-base-uncased', video_model_name='slowfast_r50', spectrogram_model_name='resnet18', device='cuda:0', all_captions_dict=None, mlp_object=None, weighted_loss_mlp_fusion=None, run_caption_model=False):

    LanguageModel_obj, VideoModel_obj, SpectrogramModel_obj = None, None, None
    in_dims_self_attention = 0
    out_embed_dim_lang, out_embed_dim_video, out_embed_dim_audio = 0,0,0
    if 'text' in modalities:
        out_embed_dim_lang = 200 if not ablation_for_caption_modality else 310
        LanguageModel_obj = LanguageModel(model_name = language_model_name, out_embed_dim = out_embed_dim_lang)
        in_dims_self_attention+=LanguageModel_obj.model.classifier.out_features
    if 'video' in modalities:
        out_embed_dim_video = 200 if not ablation_for_caption_modality else 300
        VideoModel_obj = VideoModel(model_name = video_model_name, out_embed_dim = out_embed_dim_video)
        in_dims_self_attention+=VideoModel_obj._modules['model'].blocks._modules['6'].proj.out_features
    if 'audio' in modalities:
        out_embed_dim_audio = 200 if not ablation_for_caption_modality else 300
        SpectrogramModel_obj = SpectrogramModel(model_name = spectrogram_model_name, out_embed_dim=out_embed_dim_audio)
        in_dims_self_attention+=SpectrogramModel_obj._modules['model'].fc.out_features
    if mlp_object:
        in_dims_self_attention+=mlp_object.hidden_layer_sizes[-1]
        

    batch_size = 1
    softmax = nn.Softmax(dim=1)
    if pairwise_attention_modalities:
        #Pairwise attention
        in_dims = 200
        out_dims = 1200    
    elif vanilla_fusion:
        #Baseline
        in_dims = None
        if len(modalities)==2:
            out_dims = 400
        elif len(modalities)==1:
            out_dims = 200
        else:
            out_dims = 600
    else:
        #Concate and then self-attention
        in_dims = in_dims_self_attention
        out_dims = in_dims

    intermediate_dims = 50
    self_attention = not pairwise_attention_modalities
    if not run_caption_model:
        UnifiedModel_obj = UnifiedModel(out_dims, intermediate_dims, in_dims, vanilla_fusion, self_attention, LanguageModel_obj, VideoModel_obj, SpectrogramModel_obj, mlp_object, weighted_loss_mlp_fusion).to(device)
        
        state_dict = torch.load(checkpoint_path)
        if ["start_epoch", "optimizer_state_dict", "model_state_dict", "best_loss", "random_state_dict"]==list(state_dict.keys()):
            model_state_dict = state_dict['model_state_dict']
        else:
            model_state_dict = torch.load(checkpoint_path)
        UnifiedModel_obj.load_state_dict(model_state_dict, strict=True)
        UnifiedModel_obj.eval()

    videos = pickle.load(open(videos_pkl,'rb'))
    
    val_dataset_dict = {
    'root_dir':root_dir_path,
    'all_encoded_videos':videos,
    'encoded_video_obj':EncodeVideo_obj,
    'device':device,
    'modalities':modalities,
    'all_captions_dict':all_captions_dict
    }

    val_dataset = VideoClipDataset(**val_dataset_dict)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
    classes = val_dataset.classes
    reverse_class_map = {v:k for k,v in classes.items()}

    print('\n Evaluating over:{}'.format(len(val_dataloader)))
    preds_val = list()
    targets_val = list()
    
    videos = list()
    captions = list() 
    class_from_lda = list()
    

    if run_caption_model:
        processor = AutoProcessor.from_pretrained("microsoft/git-large-vatex")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vatex")
        #sentence_transformer = SentenceTransformer("all-mpnet-base-v2") 
        csv_file = open(os.path.join(os.getcwd(),'runs',experiment_name,'lda_preds_{}.csv'.format(eval_dataset_type)), 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Video path','Caption','Prediction_from_lda'])

    for i, modality_inputs in tqdm(enumerate(val_dataloader)):
        with torch.no_grad():
            video_path, transformed_video, processed_speech, spectrogram, caption, target = modality_inputs

            if run_caption_model:
                summarized_string = summarize(video_path[0], model, processor, device)      
                csv_writer.writerow([video_path[0], summarized_string,''])#, class_from_lda[i]])
                ## embed, use lda over it and predict the class
            else:
                if not isinstance(transformed_video, int) and not isinstance(transformed_video, torch.Tensor):
                    transformed_video = [elem.to(device, non_blocking=True) for elem in transformed_video]
                if not isinstance(processed_speech, int) and not isinstance(processed_speech, torch.Tensor):
                    #pdb.set_trace()
                    processed_speech = {key:processed_speech[key].to(device, non_blocking=True) for key in processed_speech.keys()}

                spectrogram = spectrogram.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                if not isinstance(caption, int):
                    caption = caption.to(device, non_blocking=True)

                predictions_tuple = UnifiedModel_obj(processed_speech, transformed_video, spectrogram, caption)
                if isinstance(predictions_tuple, tuple):
                    predictions, _ = predictions_tuple
                else:
                    predictions = predictions_tuple
                
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
        predictions_df.to_csv(os.path.join(os.getcwd(),'runs',experiment_name,'predictions_{}.csv'.format(eval_dataset_type)), index=False)


    print(f'f1-score:{round(multiclass_f1_score(preds_val, targets_val, num_classes=2, average="micro").item()*100, 2)}')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir_path', type=str, default=os.path.join(os.path.expanduser('~'), 'cls_data_1_min'), help='path where videos will be stored in the form of root_folder/encoded_videos/<class>/video_dir/video_subclips/<video_file>')
    parser.add_argument('--run_caption_model', action='store_true', help='if set runs the caption (natural language summary) generation model for the dataset in question')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment run, a directory will be created by this name having the logs, evaluations and model weights')
    parser.add_argument('--get_classified_list', action='store_true', help='if set will save the results for each video clip in a csv file')
    parser.add_argument('--pairwise_attention_modalities', action='store_true', help='if set then late fusion will have cross modal attention instead of self attention')
    parser.add_argument('--vanilla_fusion', action='store_true', help='if set late fusion will be simple concatenation')
    parser.add_argument('--mlp_fusion', action='store_true', help='if set CaptionNet embeddings will be present for late fusion along with other modalities')
    parser.add_argument('--weighted_loss_mlp_fusion', action='store_true', help='if set loss function will be combination of the original pipeline and mlp considered separately')
    parser.add_argument('--mlp_object_path', type=str, default='', help='path to the sklearn/pytorch mlp object')
    parser.add_argument('--eval_dataset_type', type=str, help='Specify which dataset to perform evaluation on, select one out of train or val')
    parser.add_argument('--ablation_for_caption_modality', action='store_true', help='if set will carry out an ablation experiment removing caption modality, increasing the params for other three modalities')
    parser.add_argument('--modalities',nargs='+',default='video audio text',help='Add modality names out of video, audio, text')

    
    args = parser.parse_args()
    root_dir_path = args.root_dir_path
    ablation_for_caption_modality = args.ablation_for_caption_modality
    mlp_fusion = args.mlp_fusion
    mlp_object_path = args.mlp_object_path
    weighted_loss_mlp_fusion = args.weighted_loss_mlp_fusion
    run_caption_model = args.run_caption_model 
    experiment_name = args.experiment_name
    get_classified_list = args.get_classified_list
    pairwise_attention_modalities = args.pairwise_attention_modalities
    vanilla_fusion = args.vanilla_fusion
    eval_dataset_type = args.eval_dataset_type
    modalities = args.modalities.split(' ') if args.modalities=='video audio text' else args.modalities

    runs_dir = os.path.join(os.getcwd(),'runs')
    experiment_dir = os.path.join(runs_dir, experiment_name)

    mlp_object = None
    all_captions_dict = None
    if mlp_fusion:
        if os.path.splitext(mlp_object_path)[1].replace('.','')=='joblib' or os.path.splitext(mlp_object_path)[1].replace('.','')=='pkl':
            mlp_object = joblib.load(mlp_object_path)
        else:
            mlp_object = torch.load(mlp_object_path)
        all_captions_dict = pickle.load(open(os.path.join(experiment_dir,'all_captions.pkl'),'rb'))

    videos_pkl = os.path.join(experiment_dir,f'{eval_dataset_type}_encoded_video.pkl' )
    classes = {elem.split('/')[-1]:i for i, elem in enumerate(glob.glob(os.path.join(root_dir_path,'processed_data/encoded_videos/*')))}
    checkpoint_path = os.path.join(os.getcwd(),'runs',experiment_name, 'best_checkpoint.pth')
    EncodeVideo_obj = EncodeVideo()

    inference_on_val(videos_pkl, eval_dataset_type, classes, checkpoint_path, root_dir_path, EncodeVideo_obj, modalities, vanilla_fusion, pairwise_attention_modalities, get_classified_list=get_classified_list, experiment_name=experiment_name, run_caption_model=run_caption_model, mlp_object=mlp_object, weighted_loss_mlp_fusion=weighted_loss_mlp_fusion, all_captions_dict=all_captions_dict)
