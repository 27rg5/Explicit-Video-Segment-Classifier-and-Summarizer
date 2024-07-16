import os
import pdb
import yaml
import torch
import random
import joblib
import shutil
import pickle
import glob
import argparse
import numpy as np
from data_utils import *
import torch.nn as nn
from models import *
from data_utils import makedir
from torch.optim import SGD, Adam
from dataset import VideoClipDataset
from models import LanguageModel, UnifiedModel
from CaptionNet.lda_gridsearch import get_captions_and_preds_from_experiement_dir
from torch.utils.data import DataLoader
from text_utils import GetTextFromAudio, TokenizeText
from video_utils import EncodeVideo
from models import VideoModel
from data_utils import caption_files_exist
from LDA import get_corpus_from_captions
from summarizer import summarize
from transformers import AutoProcessor, AutoModelForCausalLM
#from audio_utils import GetSpectrogramFromAudio
#from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import multiclass_f1_score
import warnings
warnings.filterwarnings("ignore")

def generate_caption_dict(all_video_paths, summarizer_model, video_processor, num_train_videos):
    caption_dict = dict()
    for i, video_path in enumerate(all_video_paths):
        caption = summarize(video_path, summarizer_model, video_processor)
        caption_dict[video_path] = ('train',caption) if i<num_train_videos else ('val',caption)
    return caption_dict

def get_train_val_split_videos(root_dir, encoded_videos_path, mlp_fusion=False, split_pct=0.2):
    
    #Split explicit_train_val videos
    assert len(glob.glob(os.path.join(encoded_videos_path,'explicit/*/spectro_encs/*')))==len(glob.glob(os.path.join(encoded_videos_path,'explicit/*/audio_encs/*'))), "Number of audio and spectrogram encodings don't match for explicit videos"
    if glob.glob(os.path.join(encoded_videos_path,'explicit/*/video_subclips/*'))!=glob.glob(os.path.join(encoded_videos_path,'explicit/*/spectro_encs/*')):
        explicit_videos = [elem.replace('_spectro_enc','').replace('spectro_encs','video_subclips')+'.mp4' for elem in glob.glob(os.path.join(encoded_videos_path,'explicit/*/spectro_encs/*'))]
    else:
        explicit_videos = glob.glob(os.path.join(encoded_videos_path,'explicit/*/video_subclips/*'))
    
    explicit_indices = list(range(len(explicit_videos)))
    np.random.seed(42)
    np.random.shuffle(explicit_indices)
    explicit_val_split_index = int(len(explicit_videos)*split_pct)
    explicit_videos_val,  explicit_videos_train = [explicit_videos[index] for index in explicit_indices[:explicit_val_split_index]], [explicit_videos[index] for index in explicit_indices[explicit_val_split_index:]]


    #Split non_explicit_train_val videos
    assert len(glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/spectro_encs/*')))==len(glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/audio_encs/*'))), "Number of audio and spectrogram encodings don't match for non explicit videos"
    if glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/video_subclips/*'))!=glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/spectro_encs/*')):
        non_explicit_videos = [elem.replace('_spectro_enc','').replace('spectro_encs','video_subclips')+'.mp4' for elem in glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/spectro_encs/*'))]
    else:
        non_explicit_videos = glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/video_subclips/*'))

    non_explicit_indices = list(range(len(non_explicit_videos)))
    np.random.shuffle(non_explicit_indices)
    non_explicit_val_split_index = int(len(non_explicit_videos)*split_pct)
    non_explicit_videos_val,  non_explicit_videos_train = [non_explicit_videos[index] for index in non_explicit_indices[:non_explicit_val_split_index]], [non_explicit_videos[index] for index in non_explicit_indices[non_explicit_val_split_index:]]

    #Get the total train_val videos
    train_videos, val_videos = explicit_videos_train+non_explicit_videos_train, explicit_videos_val+non_explicit_videos_val
    #Sanity check if train and val videos are not same
    assert len(set(train_videos).intersection(set(val_videos)))==0, 'Train and Val videos have overlap'

    all_captions_dict = None

    if mlp_fusion:
        train_captions_csv, val_captions_csv, files_exist = caption_files_exist(root_dir)
        if files_exist:
            train_captions = pd.read_csv(train_captions_csv)
            train_captions['dataset_type'] = 'train'
            val_captions = pd.read_csv(val_captions_csv)
            val_captions['dataset_type'] = 'val'
        
        else:
            train_captions, val_captions, _, _ = get_captions_and_preds_from_experiement_dir(load_captions_from_exp_dir)
            # video_processor = AutoProcessor.from_pretrained("microsoft/git-large-vatex")
            # summarizer_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vatex")
            # all_videos = train_videos + val_videos
            # all_captions_dict = generate_caption_dict(all_videos, summarizer_model, video_processor, len(train_videos))
            # del video_processor, summarizer_model
        all_captions = pd.concat([train_captions, val_captions], ignore_index=True)
        all_captions_dict = dict(zip(all_captions['Video path'].values, zip(all_captions['dataset_type'].values, all_captions['Caption'].values)))

    
    assert len(train_videos)+len(val_videos)==len(all_captions_dict), 'Number of videos captioned and number of videos available don\'t match'
    
    print('Explicit train ',len(explicit_videos_train))
    print('Non_explicit train ',len(non_explicit_videos_train))
    print('Explicit val ',len(explicit_videos_val))
    print('Non_explicit val ',len(non_explicit_videos_val))
    
    return train_videos, val_videos, len(explicit_videos_train), len(non_explicit_videos_train), all_captions_dict


def train_val(**train_val_arg_dict):
    unifiedmodel_obj, optimizer, train_dataloader, val_dataloader, n_epochs, print_every, experiment_dir, loss_, bce_with_logits_loss, device, trainable_weight2 = train_val_arg_dict.values()
    writer = SummaryWriter(experiment_dir)
    train_losses = list()
    val_losses = list()
    best_loss = float('inf')
    best_f1_score = float('-inf')
    softmax = nn.Softmax(dim=1)
    n_iters_train = 0
    n_iters_val = 0
    start_epoch = 0
    if resume:
        checkpoint = torch.load(os.path.join(experiment_dir, 'best_checkpoint.pth'))
        unifiedmodel_obj.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['start_epoch']
        random.setstate(checkpoint['random_state_dict']['python_random_state'])
        np.random.set_state(checkpoint['random_state_dict']['numpy_random_state'])
        torch.set_rng_state(checkpoint['random_state_dict']['torch_random_state'])
        if device.type=='cuda':
            torch.cuda.set_rng_state(checkpoint['random_state_dict']['cuda_random_state'])
        print('Resuming training from epoch:{}'.format(start_epoch))

    
    for epoch in range(start_epoch, n_epochs):
        #train
        print('\n\n Epoch: {}'.format(epoch+1))
        print('\n Train')
        epoch_loss_train=0
        correct_train_preds = 0
        unifiedmodel_obj.train()
        preds_train = list()
        targets_train = list()
        preds_val = list()
        targets_val = list()

        for i, modality_inputs in enumerate(train_dataloader):
            _, transformed_video, processed_speech, spectrogram, caption, target = modality_inputs
            if not isinstance(transformed_video, int):
                transformed_video = [elem.to(device, non_blocking=True) for elem in transformed_video]
            if not isinstance(processed_speech, int):
                processed_speech = {key:processed_speech[key].to(device, non_blocking=True) for key in processed_speech.keys()}
            spectrogram = spectrogram.to(device, non_blocking=True)
            if not isinstance(caption, int):
                caption = caption.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            predictions_tuple = unifiedmodel_obj(processed_speech, transformed_video, spectrogram, caption)
            if isinstance(predictions_tuple, tuple) and trainable_weight2: #and trainable_weight1:
                predictions, captionnet_preds = predictions_tuple
                #positive_weight1 = torch.exp(trainable_weight1)
                positive_weight2 = torch.exp(trainable_weight2)
                batch_loss = loss_(predictions, target) + positive_weight2*bce_with_logits_loss(captionnet_preds, target.unsqueeze(0).to(torch.float32))
            else:
                predictions = predictions_tuple
                batch_loss = loss_(predictions, target)

            batch_loss.backward()
            optimizer.step()
            predictions = predictions.detach()
            target = target.detach()
            pred_softmax = softmax(predictions)
            pred_softmax = torch.argmax(pred_softmax, dim=1)
            num_correct_preds = (pred_softmax==target).sum()
            correct_train_preds+=num_correct_preds
            epoch_loss_train+=batch_loss.cpu().detach().item()
            n_iters_train+=1
            preds_train.append(pred_softmax.cpu().item())
            targets_train.append(target.cpu().item())
            
            

            if i % print_every == 0:
                print('Batch:{}, Train epoch loss average:{}'.format(i+1, epoch_loss_train/(i+1)))

        writer.add_scalar("Loss/train", epoch_loss_train/len(train_dataloader), epoch+1)
        preds_train = torch.tensor(preds_train)
        targets_train = torch.tensor(targets_train)
        f1_score_train = multiclass_f1_score(preds_train, targets_train, num_classes=2, average="macro").item()
        writer.add_scalar("F1/train", f1_score_train, epoch+1)
        average_train_loss_per_epoch = epoch_loss_train/len(train_dataloader)
        print('For epoch:{} the average train loss: {} and the accuracy: {} and F1-macro score: {}'.format(epoch+1, average_train_loss_per_epoch, correct_train_preds/len(train_dataloader), f1_score_train))
        train_losses.append(average_train_loss_per_epoch)
    

        #Val
        print('\n Val')
        unifiedmodel_obj.eval()
        epoch_loss_val=0
        correct_val_preds = 0
        for i, modality_inputs in enumerate(val_dataloader):
            with torch.no_grad():
                _, transformed_video, processed_speech,spectrogram, caption, target = modality_inputs
                if not isinstance(transformed_video, int):
                    transformed_video = [elem.to(device, non_blocking=True) for elem in transformed_video]
                if not isinstance(processed_speech, int):
                    processed_speech = {key:processed_speech[key].to(device, non_blocking=True) for key in processed_speech.keys()}

                spectrogram = spectrogram.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if not isinstance(caption, int):
                    caption = caption.to(device, non_blocking=True)

                predictions_tuple = unifiedmodel_obj(processed_speech, transformed_video, spectrogram, caption)
                
                if isinstance(predictions_tuple, tuple) and trainable_weight2: #and trainable_weight1:
                    predictions, captionnet_preds = predictions_tuple
                    #positive_weight1 = torch.exp(trainable_weight1)
                    positive_weight2 = torch.exp(trainable_weight2)
                    batch_loss = loss_(predictions, target) + positive_weight2*bce_with_logits_loss(captionnet_preds, target.unsqueeze(0).to(torch.float32))
                else:
                    predictions = predictions_tuple
                    batch_loss = loss_(predictions, target)
                    

                pred_softmax = softmax(predictions)
                pred_softmax = torch.argmax(pred_softmax, dim=1)
                num_correct_preds = (pred_softmax==target).sum()
                correct_val_preds+=num_correct_preds
                epoch_loss_val+=batch_loss.cpu().detach().item()
                n_iters_val+=1
                preds_val.append(pred_softmax.cpu().item())
                targets_val.append(target.cpu().item())
                
                

            if i % print_every == 0:
                print('Batch:{}, Val epoch loss average:{}'.format(i+1, epoch_loss_val/(i+1)))

        writer.add_scalar("Loss/val", epoch_loss_val/len(val_dataloader), epoch+1)
        preds_val = torch.tensor(preds_val)
        targets_val = torch.tensor(targets_val)
        f1_score_val = multiclass_f1_score(preds_val, targets_val, num_classes=2, average="macro").item()
        writer.add_scalar("F1/val", f1_score_val, epoch+1)
        average_val_loss_per_epoch = epoch_loss_val/len(val_dataloader)
        print('For epoch:{} the average val loss: {} and the accuracy:{} and F1-macro score: {}'.format(epoch+1, average_val_loss_per_epoch, correct_val_preds/len(val_dataloader),  f1_score_val))
        val_losses.append(average_val_loss_per_epoch)

        #Save model which has best validation loss
        if average_val_loss_per_epoch < best_loss:
            random_state_dict = {
                'python_random_state':random.getstate(),
                'numpy_random_state':np.random.get_state(),
                'torch_random_state':torch.get_rng_state(),
                'cuda_random_state':torch.cuda.get_rng_state() if device.type=='cuda' else None,
            }
            best_loss = average_val_loss_per_epoch
            checkpoint_dict = {
                'start_epoch':epoch+1,
                'optimizer_state_dict':optimizer.state_dict(),
                'model_state_dict':unifiedmodel_obj.state_dict(),
                'best_loss':best_loss,
                'random_state_dict':random_state_dict
            }
            torch.save(checkpoint_dict, os.path.join(experiment_dir, 'best_checkpoint.pth'))

        #Save model which has best validation f1-score
        # if f1_score_val > best_f1_score:
        #     best_f1_score = f1_score_val
        #     torch.save(unifiedmodel_obj.state_dict(), os.path.join(experiment_dir, 'best_checkpoint.pth'))

    writer.flush()
    writer.close()



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs',type=int, help='Total number of epochs for the model to be trained on')
    parser.add_argument('--learning_rate',type=float, help='Learning rate of the model')
    parser.add_argument('--optimizer_name',type=str, help='Type of optimizer, choose one of SGD, Adam')
    parser.add_argument('--root_dir', type=str,help='path where videos will be stored in the form of root_folder/encoded_videos/<class>/video_dir/video_subclips/<video_file>')
    parser.add_argument('--language_model_name', type=str,help='path to the fine-tuned model OR huggingface pretrained model name')
    parser.add_argument('--spectrogram_model_name', type=str,help='path to the fine-tuned model OR huggingface pretrained model name')
    parser.add_argument('--video_model_name', type=str,help='torch hub pretrained model name')
    parser.add_argument('--weighted_cross_entropy', action='store_true', help='if set applies weighted cross entropy') #Optional
    parser.add_argument('--experiment_name',type=str, help='Name of the experiment run, a directory will be created by this name having the logs, evaluations and model weights')
    parser.add_argument('--batch_size',type=int, help='Batch size for train and validation')
    parser.add_argument('--print_every',type=int, help='Indicates the number of batches after which running loss will be printed for every epoch')
    parser.add_argument('--modalities',nargs='+',default='video audio text',help='Add modality names out of video, audio, text')
    parser.add_argument('--pairwise_attention_modalities', action='store_true', help='if set then late fusion will have cross modal attention instead of self attention')
    parser.add_argument('--vanilla_fusion', action='store_true', help='if set late fusion will be simple concatenation')
    parser.add_argument('--mlp_fusion', action='store_true', help='if set CaptionNet embeddings will be present for late fusion along with other modalities')
    parser.add_argument('--weighted_loss_mlp_fusion', action='store_true', help='if set loss function will be combination of the original pipeline and mlp considered separately')
    parser.add_argument('--mlp_object_path', type=str, default='', help='path to the trained sklearn/pytorch mlp object')
    parser.add_argument('--lda_type', type=str, default='tfidf', help='type of lda, put one out of tfidf or bertopic')
    parser.add_argument('--load_captions_from_exp_dir', type=str,help='existing experiment_dir having train_val captions')
    parser.add_argument('--device',type=str,default='cuda:0', help='Use one of cuda:0, cuda:1, ....')
    parser.add_argument('--ablation_for_caption_modality', action='store_true', help='if set will carry out an ablation experiment removing caption modality, increasing the params for other three modalities')
    parser.add_argument('--resume', action='store_true', help='if set resume training from latest model checkpoint')

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cpu')
    n_epochs = args.n_epochs
    mlp_fusion = args.mlp_fusion
    mlp_object_path = args.mlp_object_path
    weighted_loss_mlp_fusion = args.weighted_loss_mlp_fusion
    learning_rate = args.learning_rate
    root_dir = args.root_dir
    lda_type = args.lda_type
    resume = args.resume
    language_model_name = args.language_model_name
    spectrogram_model_name = args.spectrogram_model_name
    video_model_name = args.video_model_name
    optimizer_name = args.optimizer_name
    print_every = args.print_every
    modalities = args.modalities.split(' ')
    ablation_for_caption_modality = args.ablation_for_caption_modality
    
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    vanilla_fusion = args.vanilla_fusion
    pairwise_attention_modalities = args.pairwise_attention_modalities

    runs_dir = os.path.join(os.getcwd(),'runs')
    experiment_dir = os.path.join(runs_dir, experiment_name)
    if os.path.exists(experiment_dir) and not resume:
        shutil.rmtree(experiment_dir)
    load_captions_from_exp_dir = os.path.join(runs_dir,args.load_captions_from_exp_dir)
    makedir(runs_dir)
    makedir(experiment_dir)

    args_dict = vars(args)
    yaml.dump(args_dict, open(os.path.join(experiment_dir,'args.yaml'),'w'), default_flow_style=False)

    weighted_cross_entropy = args.weighted_cross_entropy

    mlp_object = None
    if mlp_fusion:
        if os.path.splitext(mlp_object_path)[1].replace('.','')=='joblib' or os.path.splitext(mlp_object_path)[1].replace('.','')=='pkl':
            mlp_object = joblib.load(mlp_object_path)
        else:
            mlp_object = torch.load(mlp_object_path)
        
        
    

    ##Model init
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
        #Concatenate and then self-attention
        in_dims = in_dims_self_attention
        out_dims = in_dims

    intermediate_dims = 50
    self_attention = not pairwise_attention_modalities
    UnifiedModel_obj = UnifiedModel(out_dims, intermediate_dims, in_dims, vanilla_fusion, self_attention, LanguageModel_obj, VideoModel_obj, SpectrogramModel_obj, mlp_object, weighted_loss_mlp_fusion).to(device)

    #trainable_weight1, trainable_weight2 = None, None
    trainable_weight2 = None
    if weighted_loss_mlp_fusion:
        #trainable_weight1 = nn.Parameter(torch.empty(1).uniform_(0, 1).to(device))
        trainable_weight2 = nn.Parameter(torch.empty(1).uniform_(0, 1).to(device))
        params_for_optim = list(UnifiedModel_obj.parameters())+[trainable_weight2]#[trainable_weight1, trainable_weight2]
    else:
        params_for_optim = UnifiedModel_obj.parameters()

    if optimizer_name in ['SGD','sgd']:
        optimizer = SGD(params_for_optim, lr=learning_rate, momentum=0.9)
    elif optimizer_name in ['Adam','adam']:
        optimizer = Adam(params_for_optim, lr=learning_rate)

    encoded_videos_path = os.path.join(root_dir,'encoded_videos')
    
    EncodeVideo_obj = EncodeVideo()

    if not os.path.exists(encoded_videos_path):
        all_videos = glob.glob(os.path.join(root_dir,'non_encoded_videos_sep/*/*'))
        GetTextFromAudio_obj = GetTextFromAudio()
        TokenizeText_obj = TokenizeText()        
        encode_videos(all_videos, encoded_videos_path, EncodeVideo_obj, GetTextFromAudio_obj, TokenizeText_obj)    
    
    
    train_encoded_videos, val_encoded_videos, num_explicit_videos_train, num_non_explicit_videos_train, all_captions_dict = get_train_val_split_videos(root_dir, encoded_videos_path, mlp_fusion=mlp_fusion)
    all_captions_dict = get_corpus_from_captions(all_captions_dict, lda_type) if all_captions_dict else None

    pickle.dump(val_encoded_videos, open(os.path.join(experiment_dir,'val_encoded_video.pkl'), 'wb'))
    print('Val videos stored')
    pickle.dump(train_encoded_videos, open(os.path.join(experiment_dir,'train_encoded_video.pkl'), 'wb'))
    print('Train videos stored')

    if mlp_fusion:
        if not os.path.exists(os.path.join(experiment_dir,'all_captions.pkl')):
            pickle.dump(all_captions_dict, open(os.path.join(experiment_dir,'all_captions.pkl'), 'wb'))
        else:
            all_captions_dict = pickle.load(open(os.path.join(experiment_dir,'all_captions.pkl'),'rb'))
    
    train_dataset_dict = {
        'root_dir':root_dir,
        'all_encoded_videos':train_encoded_videos,
        'encoded_video_obj':EncodeVideo_obj,
        'device':device,
        'modalities':modalities,
        'all_captions_dict':all_captions_dict
    }

    val_dataset_dict = {
        'root_dir':root_dir,
        'all_encoded_videos':val_encoded_videos,
        'encoded_video_obj':EncodeVideo_obj,
        'device':device,
        'modalities':modalities,
        'all_captions_dict':all_captions_dict
    }


    train_dataloader, val_dataloader = DataLoader(VideoClipDataset(**train_dataset_dict), shuffle=True, batch_size=batch_size, pin_memory=True),\
    DataLoader(VideoClipDataset(**val_dataset_dict), shuffle=False, batch_size=batch_size, pin_memory=True)
    if weighted_cross_entropy:
        #pdb.set_trace()
        total_videos = num_explicit_videos_train + num_non_explicit_videos_train
        class_dist = [num_explicit_videos_train, num_non_explicit_videos_train]
        class_weights = [1-(elem/total_videos) for elem in class_dist]
        loss_ = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        loss_ = nn.CrossEntropyLoss()

    bce_with_logits_loss = None
    if weighted_loss_mlp_fusion:
        bce_with_logits_loss = nn.BCEWithLogitsLoss()
    print('Training on \n train:{} batches \n val:{} batches'.format(len(train_dataloader), len(val_dataloader)))

    train_val_arg_dict = {
        'unifiedmodel_obj':UnifiedModel_obj, 
        'optimizer':optimizer,
        'train_dataloader':train_dataloader,
        'val_dataloader':val_dataloader,
        'n_epochs':n_epochs,
        'print_every':print_every,
        'experiment_path':experiment_dir,
        'loss':loss_,
        'bce_with_logits_loss':bce_with_logits_loss,
        'device':device,
        #'trainable_weight1':trainable_weight1,
        'trainable_weight2':trainable_weight2
    }
    train_val(**train_val_arg_dict)




