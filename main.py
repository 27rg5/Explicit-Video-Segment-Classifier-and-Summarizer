import os
import pdb
import time
import yaml
import torch
import random
import joblib
import shutil
import pickle
import glob
import argparse
import transformers
import numpy as np
from torch.utils.data import WeightedRandomSampler
from dataset import collate_fn
#from data_utils import *
import torch.nn as nn
from models import *
from torch.optim import SGD, Adam
from dataset import VideoClipDataset
from models import LanguageModel, UnifiedModel
from CaptionNet.lda_gridsearch import get_captions_and_targets_from_experiement_dir, prepare_data
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torcheval.metrics.functional import multiclass_f1_score
import warnings
warnings.filterwarnings("ignore")

#torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark_limit = 0
def generate_caption_dict(all_video_paths, summarizer_model, video_processor, num_train_videos):
    caption_dict = dict()
    for i, video_path in enumerate(all_video_paths):
        caption = summarize(video_path, summarizer_model, video_processor)
        caption_dict[video_path] = ('train',caption) if i<num_train_videos else ('val',caption)
    return caption_dict

def get_train_val_split(train_videos_pkl, val_videos_pkl):
    train_videos = pickle.load(open(train_videos_pkl,'rb'))
    val_videos = pickle.load(open(val_videos_pkl,'rb'))
    return train_videos, val_videos

# def get_train_val_split_videos(root_dir, encoded_videos_path, mlp_fusion=False, split_pct=0.2):
    
#     #Split explicit_train_val videos
#     assert len(glob.glob(os.path.join(encoded_videos_path,'explicit/*/spectro_encs/*')))==len(glob.glob(os.path.join(encoded_videos_path,'explicit/*/audio_encs/*'))), "Number of audio and spectrogram encodings don't match for explicit videos"
#     if glob.glob(os.path.join(encoded_videos_path,'explicit/*/video_subclips/*'))!=glob.glob(os.path.join(encoded_videos_path,'explicit/*/spectro_encs/*')):
#         explicit_videos = [elem.replace('_spectro_enc','').replace('spectro_encs','video_subclips')+'.mp4' for elem in glob.glob(os.path.join(encoded_videos_path,'explicit/*/spectro_encs/*'))]
#     else:
#         explicit_videos = glob.glob(os.path.join(encoded_videos_path,'explicit/*/video_subclips/*'))
    
#     explicit_indices = list(range(len(explicit_videos)))
#     np.random.seed(42)
#     np.random.shuffle(explicit_indices)
#     explicit_val_split_index = int(len(explicit_videos)*split_pct)
#     explicit_videos_val,  explicit_videos_train = [explicit_videos[index] for index in explicit_indices[:explicit_val_split_index]], [explicit_videos[index] for index in explicit_indices[explicit_val_split_index:]]


#     #Split non_explicit_train_val videos
#     assert len(glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/spectro_encs/*')))==len(glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/audio_encs/*'))), "Number of audio and spectrogram encodings don't match for non explicit videos"
#     if glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/video_subclips/*'))!=glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/spectro_encs/*')):
#         non_explicit_videos = [elem.replace('_spectro_enc','').replace('spectro_encs','video_subclips')+'.mp4' for elem in glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/spectro_encs/*'))]
#     else:
#         non_explicit_videos = glob.glob(os.path.join(encoded_videos_path,'non_explicit/*/video_subclips/*'))

#     non_explicit_indices = list(range(len(non_explicit_videos)))
#     np.random.shuffle(non_explicit_indices)
#     non_explicit_val_split_index = int(len(non_explicit_videos)*split_pct)
#     non_explicit_videos_val,  non_explicit_videos_train = [non_explicit_videos[index] for index in non_explicit_indices[:non_explicit_val_split_index]], [non_explicit_videos[index] for index in non_explicit_indices[non_explicit_val_split_index:]]

#     #Get the total train_val videos
#     train_videos, val_videos = explicit_videos_train+non_explicit_videos_train, explicit_videos_val+non_explicit_videos_val
#     #Sanity check if train and val videos are not same
#     assert len(set(train_videos).intersection(set(val_videos)))==0, 'Train and Val videos have overlap'

#     all_captions_dict = None

#     if mlp_fusion:
#         train_captions_csv, val_captions_csv, files_exist = caption_files_exist(root_dir)
#         if files_exist:
#             train_captions = pd.read_csv(train_captions_csv)
#             train_captions['dataset_type'] = 'train'
#             val_captions = pd.read_csv(val_captions_csv)
#             val_captions['dataset_type'] = 'val'
        
#         else:
#             train_captions, val_captions, _, _ = get_captions_and_preds_from_experiement_dir(load_captions_from_exp_dir)
#             # video_processor = AutoProcessor.from_pretrained("microsoft/git-large-vatex")
#             # summarizer_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vatex")
#             # all_videos = train_videos + val_videos
#             # all_captions_dict = generate_caption_dict(all_videos, summarizer_model, video_processor, len(train_videos))
#             # del video_processor, summarizer_model
#         all_captions = pd.concat([train_captions, val_captions], ignore_index=True)
#         all_captions_dict = dict(zip(all_captions['Video path'].values, zip(all_captions['dataset_type'].values, all_captions['Caption'].values)))

    
#     assert len(train_videos)+len(val_videos)==len(all_captions_dict), 'Number of videos captioned and number of videos available don\'t match'
    
    # print('Explicit train ',len(explicit_videos_train))
    # print('Non_explicit train ',len(non_explicit_videos_train))
    # print('Explicit val ',len(explicit_videos_val))
    # print('Non_explicit val ',len(non_explicit_videos_val))
    
    # return train_videos, val_videos, len(explicit_videos_train), len(non_explicit_videos_train), all_captions_dict


def train_val(**train_val_arg_dict):
    unifiedmodel_obj, optimizer, train_dataloader, val_dataloader, test_dataloader, n_epochs, print_every, experiment_dir, loss_, bce_with_logits_loss, device, use_lr_scheduler, trainable_weight2 = train_val_arg_dict.values()
    prev_time = time.time()
    writer = SummaryWriter(experiment_dir)
    train_losses = list()
    val_losses = list()
    test_losses = list()
    best_loss = float('inf')
    best_f1_score = float('-inf')
    softmax = nn.Softmax(dim=-1)
    n_iters_train = 0
    n_iters_val = 0
    n_iters_test = 0
    start_epoch = 0
    scheduler = None
    patience = 7 #Defines the number of epochs to wait for improvement before stopping early
    patience_counter = 0 #Maintains the count of epochs since last improvement

    if resume:
        checkpoint = torch.load(os.path.join(experiment_dir, 'best_checkpoint.pth'))
        unifiedmodel_obj.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['start_epoch']
        scheduler = checkpoint['scheduler_state_dict']
        random.setstate(checkpoint['random_state_dict']['python_random_state'])
        np.random.set_state(checkpoint['random_state_dict']['numpy_random_state'])
        torch.set_rng_state(checkpoint['random_state_dict']['torch_random_state'])
        if device.type=='cuda':
            torch.cuda.set_rng_state(checkpoint['random_state_dict']['cuda_random_state'])
        print('Resuming training from epoch:{}'.format(start_epoch))

    
    if use_lr_scheduler and not resume:
        #scheduler = ReduceLROnPlateau(optimizer, patience=2)
        scheduler = OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_dataloader), epochs=n_epochs)

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
        preds_test = list()
        targets_test = list()


        for i, modality_inputs in enumerate(train_dataloader):
            _, transformed_video, processed_speech, spectrogram, caption, target = modality_inputs
            # if i==1:
            #     print(f'video tensor is :{transformed_video[0].is_pinned()} {transformed_video[1].is_pinned()}\n \
            #     caption tensor is :{caption.is_pinned()} \n \
            #     target tensor is :{target.is_pinned()}')
            #     #break
            #     return

            #Breakpoint
            #pdb.set_trace()
            if isinstance(transformed_video, list):
                transformed_video = [elem.to(device, non_blocking=True) for elem in transformed_video]
            if not isinstance (processed_speech, torch.Tensor):
                processed_speech = {key:processed_speech[key].to(device, non_blocking=True) for key in processed_speech.keys()}
            if spectrogram.ndim>1:#torch.equal(spectrogram, torch.zeros_like(spectrogram)):
                spectrogram = spectrogram.to(device, non_blocking=True)
            if caption.ndim>1:#torch.equal(caption, torch.zeros_like(caption)):
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
                #Breakpoint
                #pdb.set_trace()                
                batch_loss = loss_(predictions, target)

            batch_loss.backward()
            optimizer.step()
            predictions = predictions.detach()
            target = target.detach()
            pred_softmax = softmax(predictions)
            pred_softmax = torch.argmax(pred_softmax, dim=-1)
            num_correct_preds = (pred_softmax==target).sum()
            correct_train_preds+=num_correct_preds
            epoch_loss_train+=batch_loss.cpu().detach().item()
            n_iters_train+=1
            preds_train.extend(pred_softmax.cpu().tolist())
            targets_train.extend(target.cpu().tolist())
            scheduler.step()
            
            # preds_train.append(pred_softmax.cpu())
            # targets_train.append(target.cpu())
            

            if i % print_every == 0:
                curr_time = time.time()
                minutes = int(curr_time - prev_time)//60
                seconds = ((curr_time - prev_time) - int(curr_time - prev_time))*60
                print('Batch:{}, Train epoch loss average:{} time delta in minutes:{} seconds:{}'.format(i+1, epoch_loss_train/(i+1), minutes, seconds))
                prev_time = time.time()


        writer.add_scalar("Loss/train", epoch_loss_train/len(train_dataloader), epoch+1)
        preds_train = torch.tensor(preds_train)
        targets_train = torch.tensor(targets_train)
        f1_score_train = multiclass_f1_score(preds_train, targets_train, num_classes=2, average="micro").item()
        writer.add_scalar("F1/train", f1_score_train, epoch+1)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], epoch+1)
        average_train_loss_per_epoch = epoch_loss_train/len(train_dataloader)
        print('For epoch:{} the average train loss: {} and the accuracy: {} and F1-micro score: {}'.format(epoch+1, average_train_loss_per_epoch, correct_train_preds/train_dataloader.dataset.__len__(), f1_score_train))
        train_losses.append(average_train_loss_per_epoch)
    

        #Val
        print('\n Val')
        unifiedmodel_obj.eval()
        epoch_loss_val=0
        correct_val_preds = 0
        for i, modality_inputs in enumerate(val_dataloader):
            with torch.no_grad():
                _, transformed_video, processed_speech,spectrogram, caption, target = modality_inputs
                #Breakpoint
                if isinstance(transformed_video, list):
                    transformed_video = [elem.to(device, non_blocking=True) for elem in transformed_video]
                if not isinstance (processed_speech, torch.Tensor):
                    processed_speech = {key:processed_speech[key].to(device, non_blocking=True) for key in processed_speech.keys()}
                if spectrogram.ndim>1:#torch.equal(spectrogram, torch.zeros_like(spectrogram)):
                    spectrogram = spectrogram.to(device, non_blocking=True)
                if caption.ndim>1:#torch.equal(caption, torch.zeros_like(caption)):
                    caption = caption.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

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
                pred_softmax = torch.argmax(pred_softmax, dim=-1)
                num_correct_preds = (pred_softmax==target).sum()
                correct_val_preds+=num_correct_preds
                epoch_loss_val+=batch_loss.cpu().detach().item()
                n_iters_val+=1
                preds_val.extend(pred_softmax.cpu().tolist())
                targets_val.extend(target.cpu().tolist())
                
                

            if i % print_every == 0:
                print('Batch:{}, Val epoch loss average:{}'.format(i+1, epoch_loss_val/(i+1)))

        writer.add_scalar("Loss/val", epoch_loss_val/len(val_dataloader), epoch+1)
        preds_val = torch.tensor(preds_val)
        targets_val = torch.tensor(targets_val)
        f1_score_val = multiclass_f1_score(preds_val, targets_val, num_classes=2, average="micro").item()
        writer.add_scalar("F1/val", f1_score_val, epoch+1)
        average_val_loss_per_epoch = epoch_loss_val/len(val_dataloader)
        # if use_lr_scheduler:
        #     scheduler.step(average_val_loss_per_epoch)

        print('For epoch:{} the average val loss: {} and the accuracy:{} and F1-micro score: {}'.format(epoch+1, average_val_loss_per_epoch, correct_val_preds/val_dataloader.dataset.__len__(),  f1_score_val))
        val_losses.append(average_val_loss_per_epoch)

        #Save model which has best validation loss
        if average_val_loss_per_epoch < best_loss:
            patience_counter = 0
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
#                'model_state_dict':unifiedmodel_obj.module.state_dict(),
                'model_state_dict':unifiedmodel_obj.state_dict(),
                'best_loss':best_loss,
                'random_state_dict':random_state_dict,
                'scheduler_state_dict':scheduler.state_dict() if scheduler else None
            }
            torch.save(checkpoint_dict, os.path.join(experiment_dir, 'best_checkpoint.pth'))
        else:
            patience_counter+=1
        
        #Save model which has best validation f1-score
        # if f1_score_val > best_f1_score:
        #     best_f1_score = f1_score_val
        #     torch.save(unifiedmodel_obj.state_dict(), os.path.join(experiment_dir, 'best_checkpoint.pth'))

        print('\n Test')
        unifiedmodel_obj.eval()
        epoch_loss_test=0
        correct_test_preds = 0
        for i, modality_inputs in enumerate(test_dataloader):
            with torch.no_grad():
                _, transformed_video, processed_speech,spectrogram, caption, target = modality_inputs
                #Breakpoint
                if isinstance(transformed_video, list):
                    transformed_video = [elem.to(device, non_blocking=True) for elem in transformed_video]
                if not isinstance (processed_speech, torch.Tensor):
                    processed_speech = {key:processed_speech[key].to(device, non_blocking=True) for key in processed_speech.keys()}
                if spectrogram.ndim>1:#torch.equal(spectrogram, torch.zeros_like(spectrogram)):
                    spectrogram = spectrogram.to(device, non_blocking=True)
                if caption.ndim>1:#torch.equal(caption, torch.zeros_like(caption)):
                    caption = caption.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

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
                pred_softmax = torch.argmax(pred_softmax, dim=-1)
                num_correct_preds = (pred_softmax==target).sum()
                correct_test_preds+=num_correct_preds
                epoch_loss_test+=batch_loss.cpu().detach().item()
                n_iters_test+=1
                preds_test.extend(pred_softmax.cpu().tolist())
                targets_test.extend(target.cpu().tolist())
                
                

            if i % print_every == 0:
                print('Batch:{}, Test epoch loss average:{}'.format(i+1, epoch_loss_test/(i+1)))

        writer.add_scalar("Loss/Test", epoch_loss_test/len(test_dataloader), epoch+1)
        preds_test = torch.tensor(preds_test)
        targets_test = torch.tensor(targets_test)
        f1_score_test = multiclass_f1_score(preds_test, targets_test, num_classes=2, average="micro").item()
        writer.add_scalar("F1/Test", f1_score_test, epoch+1)
        average_test_loss_per_epoch = epoch_loss_test/len(test_dataloader)
        # if use_lr_scheduler:
        #     scheduler.step(average_val_loss_per_epoch)

        print('For epoch:{} the average test loss: {} and the accuracy:{} and F1-micro score: {}'.format(epoch+1, average_test_loss_per_epoch, correct_test_preds/test_dataloader.dataset.__len__(),  f1_score_test))
        test_losses.append(average_test_loss_per_epoch)

        if patience_counter>=patience:
            print('Early stopping at epoch:{}, quitting the program....'.format(epoch+1))
            break

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
    parser.add_argument('--modalities',nargs='+',default=['video','audio', 'text'],help='Add modality names out of video, audio, text')
    parser.add_argument('--pairwise_attention_modalities', action='store_true', help='if set then late fusion will have cross modal attention instead of self attention')
    parser.add_argument('--vanilla_fusion', action='store_true', help='if set late fusion will be simple concatenation')
    parser.add_argument('--mlp_fusion', action='store_true', help='if set CaptionNet embeddings will be present for late fusion along with other modalities')
    parser.add_argument('--weighted_loss_mlp_fusion', action='store_true', help='if set loss function will be combination of the original pipeline and mlp considered separately')
    parser.add_argument('--mlp_object_path', type=str, default='', help='path to the trained sklearn/pytorch mlp object')
    parser.add_argument('--lda_type', type=str, default='tfidf', help='type of lda, put one out of tfidf or bertopic')
    parser.add_argument('--captions_data_names_pkl_path', type=str,help='existing experiment_dir having train_val captions')
    parser.add_argument('--device',type=str,default='cuda:0', help='Use one of cuda:0, cuda:1, ....')
    parser.add_argument('--ablation_for_caption_modality', action='store_true', help='if set will carry out an ablation experiment removing caption modality, increasing the params for other three modalities')
    parser.add_argument('--resume', action='store_true', help='if set resume training from latest model checkpoint')
    parser.add_argument('--weight_decay',type=float, default=0, help='set weight decay param for the optimizer of choice')
    parser.add_argument('--use_lr_scheduler', action='store_true',help='if set uses onecyclelr')
    parser.add_argument('--dropout', type=float, default=0,help='denotes the value of dropout used after modality fusion')

    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"- {arg}: {value}")

    device = torch.device(args.device) if args.device else torch.device('cpu')
    weight_decay = args.weight_decay
    use_lr_scheduler = args.use_lr_scheduler
    dropout = args.dropout
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
    modalities = args.modalities
    ablation_for_caption_modality = args.ablation_for_caption_modality
    
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    vanilla_fusion = args.vanilla_fusion
    pairwise_attention_modalities = args.pairwise_attention_modalities

    runs_dir = os.path.join(os.getcwd(),'runs')
    experiment_dir = os.path.join(runs_dir, experiment_name)
    if os.path.exists(experiment_dir) and not resume:
        shutil.rmtree(experiment_dir)
    captions_data_names_pkl_path = args.captions_data_names_pkl_path
    # makedir(runs_dir)
    # makedir(experiment_dir)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

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
    dims_for_late_fusion = 0
    out_embed_dim_lang, out_embed_dim_video, out_embed_dim_audio = 0,0,0
    modality_out_dim_mapping = dict()
    #Breakpoint
    #pdb.set_trace()
    
    if 'text' in modalities:
        out_embed_dim_lang = 200 if not ablation_for_caption_modality else 310
        LanguageModel_obj = LanguageModel(model_name = language_model_name, out_embed_dim = out_embed_dim_lang)
        modality_out_dim_mapping['text'] = LanguageModel_obj.model.classifier.out_features
        dims_for_late_fusion+=LanguageModel_obj.model.classifier.out_features
    if 'video' in modalities:
        out_embed_dim_video = 200 if not ablation_for_caption_modality else 300
        VideoModel_obj = VideoModel(model_name = video_model_name, out_embed_dim = out_embed_dim_video)
        modality_out_dim_mapping['video'] = VideoModel_obj._modules['model'].blocks._modules['6'].proj.out_features
        dims_for_late_fusion+=VideoModel_obj._modules['model'].blocks._modules['6'].proj.out_features
    if 'audio' in modalities:
        out_embed_dim_audio = 200 if not ablation_for_caption_modality else 300
        SpectrogramModel_obj = SpectrogramModel(model_name = spectrogram_model_name, out_embed_dim=out_embed_dim_audio)
        modality_out_dim_mapping['audio'] = SpectrogramModel_obj._modules['model'].fc.out_features
        dims_for_late_fusion+=SpectrogramModel_obj._modules['model'].fc.out_features
    if mlp_object:
        modality_out_dim_mapping['caption'] = mlp_object.hidden_layer_sizes[-1]
        dims_for_late_fusion+=mlp_object.hidden_layer_sizes[-1]
    
    if pairwise_attention_modalities:
        #Pairwise attention
        in_dims = 200
        out_dims = 1200    
    elif vanilla_fusion:
        #Baseline
        modality_out_dim_mapping = dict()
        in_dims = None
        out_dims = dims_for_late_fusion
    else:
        #Concatenate and then self-attention
        in_dims = out_dims = dims_for_late_fusion

    intermediate_dims = 50
    self_attention = not pairwise_attention_modalities
    UnifiedModel_obj = UnifiedModel(out_dims, intermediate_dims, in_dims, modality_out_dim_mapping, dropout, vanilla_fusion, self_attention, LanguageModel_obj, VideoModel_obj, SpectrogramModel_obj, mlp_object, weighted_loss_mlp_fusion).to(device)
    #Breakpoint
    #pdb.set_trace()
    # num_devices = 4
    # UnifiedModel_obj = torch.nn.DataParallel(UnifiedModel_obj, device_ids = [id for id in range(num_devices)]).to(device)

    #trainable_weight1, trainable_weight2 = None, None
    trainable_weight2 = None
    if weighted_loss_mlp_fusion:
        #trainable_weight1 = nn.Parameter(torch.empty(1).uniform_(0, 1).to(device))
        trainable_weight2 = nn.Parameter(torch.empty(1).uniform_(0, 1).to(device))
        params_for_optim = list(UnifiedModel_obj.parameters())+[trainable_weight2]#[trainable_weight1, trainable_weight2]
    else:
        params_for_optim = UnifiedModel_obj.parameters()

    if optimizer_name in ['SGD','sgd']:
        optimizer = SGD(params_for_optim, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name in ['Adam','adam']:
        optimizer = Adam(params_for_optim, weight_decay=weight_decay, lr=learning_rate)

    encoded_videos_path = os.path.join(root_dir,'encoded_videos')
    
    EncodeVideo_obj_train = EncodeVideo()
    EncodeVideo_obj_val = EncodeVideo(mode='val')
    EncodeVideo_obj_test = EncodeVideo(mode='test')
    
    all_captions_dict = None
    #train_encoded_videos, val_encoded_videos, num_explicit_videos_train, num_non_explicit_videos_train, all_captions_dict = get_train_val_split_videos(root_dir, encoded_videos_path, mlp_fusion=mlp_fusion)
    assert os.path.exists(captions_data_names_pkl_path), 'Experiment directory with captions doesn\'t exist'
    train_videos_path, val_videos_path, test_videos_path = os.path.join(captions_data_names_pkl_path,'train_val_test_videos_pkl/train_videos.pkl'), \
    os.path.join(captions_data_names_pkl_path,'train_val_test_videos_pkl/val_videos.pkl'), \
    os.path.join(captions_data_names_pkl_path,'train_val_test_videos_pkl/test_videos.pkl')

    train_encoded_videos = pickle.load(open(train_videos_path,'rb'))
    val_encoded_videos = pickle.load(open(val_videos_path,'rb'))
    test_encoded_videos = pickle.load(open(test_videos_path,'rb'))
    captions_path = os.path.join(captions_data_names_pkl_path,'captions/all_captions.pkl')
    
    if os.path.exists(captions_path):
        all_captions_dict = pickle.load(open(captions_path,'rb'))
    else:
        train_captions_df, val_captions_df, test_captions_df,_,_,_ = get_captions_and_targets_from_experiement_dir(captions_data_names_pkl_path)
        all_captions_dict, _ = prepare_data(train_captions_df, val_captions_df, test_captions_df, lda_type='bertopic')
        pickle.dump(all_captions_dict, open(captions_path,'wb'))

    # pickle.dump(val_encoded_videos, open(os.path.join(experiment_dir,'val_encoded_video.pkl'), 'wb'))
    # print('Val videos stored')
    # pickle.dump(train_encoded_videos, open(os.path.join(experiment_dir,'train_encoded_video.pkl'), 'wb'))
    # print('Train videos stored')

    if mlp_fusion:
        all_captions_dict = pickle.load(open(captions_path,'rb'))    

    train_dataset_dict = {
        'root_dir':root_dir,
        'all_encoded_videos':train_encoded_videos,
        'encoded_video_obj':EncodeVideo_obj_train,
        'modalities':modalities,
        'all_captions_dict':all_captions_dict
    }

    val_dataset_dict = {
        'root_dir':root_dir,
        'all_encoded_videos':val_encoded_videos,
        'encoded_video_obj':EncodeVideo_obj_val,
        'modalities':modalities,
        'all_captions_dict':all_captions_dict
    }

    test_dataset_dict = {
        'root_dir':root_dir,
        'all_encoded_videos':test_encoded_videos,
        'encoded_video_obj':EncodeVideo_obj_test,
        'modalities':modalities,
        'all_captions_dict':all_captions_dict
    }


    train_dataset = VideoClipDataset(**train_dataset_dict)
    val_dataset = VideoClipDataset(**val_dataset_dict)
    test_dataset = VideoClipDataset(**test_dataset_dict)

    labels_np = np.array(train_dataset.labels)
    class_counts = np.bincount(labels_np)
    class_weights = 1.0 / class_counts  # Inverse frequency

    # Create a mapping from class ID to class weight
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    # Assign a weight to each sample in the dataset
    sample_weights = [class_weight_dict[label] for label in train_dataset.labels]
    
    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # You can adjust this number if needed
        replacement=True,
        generator=torch.Generator().manual_seed(42)  # Set to True to allow sampling with replacement
    )
    
    train_dataloader, val_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn, num_workers=10),\
    DataLoader(val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True,  collate_fn=collate_fn, num_workers=10), DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True,  collate_fn=collate_fn, num_workers=10)
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
    print('Training on \n train:{} batches \n val:{} batches \n test:{} batches'.format(len(train_dataloader), len(val_dataloader), len(test_dataloader)))

    train_val_arg_dict = {
        'unifiedmodel_obj':UnifiedModel_obj, 
        'optimizer':optimizer,
        'train_dataloader':train_dataloader,
        'val_dataloader':val_dataloader,
        'test_dataloader':test_dataloader,
        'n_epochs':n_epochs,
        'print_every':print_every,
        'experiment_path':experiment_dir,
        'loss':loss_,
        'bce_with_logits_loss':bce_with_logits_loss,
        'device':device,
        'use_lr_scheduler':use_lr_scheduler,
        #'trainable_weight1':trainable_weight1,
        'trainable_weight2':trainable_weight2
    }
    train_val(**train_val_arg_dict)




