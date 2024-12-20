#TRAIN ABLATIONS WITH SELF ATTENTION
#Video Audio Text Caption
echo "TRAINING VIDEO AUDIO TEXT CAPTION"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --language_model_name  distilbert-base-uncased  --experiment_name final_pipeline_v1 --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

#Audio Text
echo -e "\nTRAINING AUDIO TEXT"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased  --experiment_name final_pipeline_audio_text --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities text audio

#Video Audio
echo -e "\nTRAINING VIDEO AUDIO"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_audio --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video audio

#Video Text
echo -e "\nTRAINING VIDEO TEXT"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased --video_model_name slowfast_r50 --experiment_name final_pipeline_video_text --batch_size 5 --print_every  10  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video text

#Video
echo -e "\nTRAINING VIDEO"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video --batch_size 5 --print_every 10  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video

#Audio
echo -e "\nTRAINING AUDIO"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --experiment_name final_pipeline_audio --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities audio

#Text
echo -e "\nTRAINING TEXT"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased --experiment_name final_pipeline_text --batch_size 5 --print_every  10  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities text


#TRAIN ABLATIONS WITH SIMPLE CONCATENATION
#Audio Text Vanilla
echo "TRAINING AUDIO TEXT VANILLA"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased  --experiment_name final_pipeline_audio_text_vanilla --vanilla_fusion --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities text audio

#Video Audio Vanilla
echo -e "\nTRAINING VIDEO AUDIO VANILLA"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_audio_vanilla --vanilla_fusion --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video audio

#Video Text
echo -e "\nTRAINING VIDEO TEXT VANILLA"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased --video_model_name slowfast_r50  --experiment_name final_pipeline_video_text_vanilla --vanilla_fusion --batch_size 5 --print_every  10  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video text

#Video
echo -e "\nTRAINING VIDEO VANILLA"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_vanilla --vanilla_fusion --batch_size 5 --print_every 10  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video

#Audio
echo -e "\nTRAINING AUDIO VANILLA"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --experiment_name final_pipeline_audio_vanilla --vanilla_fusion --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities audio

#Text
echo -e "\nTRAINING TEXT VANILLA"
python -W ignore main.py --n_epochs 30 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased --experiment_name final_pipeline_text_vanilla --vanilla_fusion --batch_size 5 --print_every  10  --device cuda:0 --exp_dir_with_captions attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities text
