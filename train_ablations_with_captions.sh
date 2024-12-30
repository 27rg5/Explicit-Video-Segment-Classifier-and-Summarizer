#TRAIN ABLATIONS WITH SELF ATTENTION
#Video Audio Text Caption
echo "TRAINING VIDEO AUDIO TEXT CAPTION"
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --language_model_name  distilbert-base-uncased  --experiment_name final_pipeline_v1 --batch_size 5 --print_every 10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Audio Text
echo -e "\nTRAINING AUDIO TEXT CAPTION "
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased  --experiment_name final_pipeline_audio_text_caption --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities text audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Video Audio
echo -e "\nTRAINING VIDEO AUDIO CAPTION "
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_audio_caption --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Video Text
echo -e "\nTRAINING VIDEO TEXT CAPTION "
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased --video_model_name slowfast_r50 --experiment_name final_pipeline_video_text_caption --batch_size 5 --print_every  10  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video text --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Video
echo -e "\nTRAINING VIDEO CAPTION "
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_caption --batch_size 5 --print_every 10  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Audio
echo -e "\nTRAINING AUDIO CAPTION "
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --experiment_name final_pipeline_audio_caption --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Text
echo -e "\nTRAINING TEXT CAPTION "
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased --experiment_name final_pipeline_text_caption --batch_size 5 --print_every  10  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities text --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#TRAIN ABLATIONS WITH SIMPLE CONCATENATION

#Video Audio Text Caption Vanilla
echo -e "\nTRAINING VIDEO AUDIO TEXT CAPTION VANILLA"
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --language_model_name  distilbert-base-uncased  --experiment_name final_pipeline_v1_vanilla --vanilla_fusion --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Audio Text Vanilla
echo -e "\nTRAINING AUDIO TEXT CAPTION VANILLA"
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased  --experiment_name final_pipeline_audio_text_caption_vanilla --vanilla_fusion --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities text audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Video Audio Vanilla
echo -e "\nTRAINING VIDEO AUDIO CAPTION VANILLA"
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_audio_caption_vanilla --vanilla_fusion --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Video Text
echo -e "\nTRAINING VIDEO TEXT CAPTION VANILLA"
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased --video_model_name slowfast_r50  --experiment_name final_pipeline_video_text_caption_vanilla --vanilla_fusion --batch_size 5 --print_every  10  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video text --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Video
echo -e "\nTRAINING VIDEO CAPTION VANILLA"
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_caption_vanilla --vanilla_fusion --batch_size 5 --print_every 10  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Audio
echo -e "\nTRAINING AUDIO CAPTION VANILLA"
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --experiment_name final_pipeline_audio_caption_vanilla --vanilla_fusion --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

#Text
echo -e "\nTRAINING TEXT CAPTION VANILLA"
python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --language_model_name  distilbert-base-uncased --experiment_name final_pipeline_text_caption_vanilla --vanilla_fusion --batch_size 5 --print_every  10  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities text --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2


##################################################### REGULARIZATION EXPERIMENTS ##################################################
#Video Audio Vanilla with wd 1e-4
# echo -e "\nTRAINING VIDEO AUDIO CAPTION WD 1E-4 VANILLA"
# python -W ignore main.py --n_epochs 50 --learning_rate 1e-3 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_audio_caption_wd_1e_4_vanilla --vanilla_fusion --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4

# echo -e "\nTRAINING VIDEO AUDIO CAPTION WD 1E-4 LR 1E-4 DROP 0.3 VANILLA"
# python -W ignore main.py --n_epochs 50 --learning_rate 1e-4 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_audio_caption_wd_drop_lr_sched_vanilla --vanilla_fusion --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --dropout 0.3 --use_lr_scheduler

# echo -e "\nTRAINING VIDEO AUDIO CAPTION WD 1E-4 LR 2E-4 DROP 0.2 VANILLA"
# python -W ignore main.py --n_epochs 50 --learning_rate 2e-4 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_audio_caption_wd_drop_lr_sched_vanilla_v2 --vanilla_fusion --batch_size 5 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

# echo -e "\nTRAINING VIDEO AUDIO CAPTION WD 1E-4 LR 2E-4 DROP 0.2 AUGMENTATIONS VANILLA"
# python -W ignore main.py --n_epochs 50 --learning_rate 2e-4 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_audio_caption_wd_drop_lr_sched_vanilla_v3 --vanilla_fusion --batch_size 7 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2

# echo -e "\nTRAINING VIDEO AUDIO CAPTION WD 1E-4 LR 1E-5 DROP 0.2 AUGMENTATIONS ONECYCLE VANILLA"
# python -W ignore main.py --n_epochs 50 --learning_rate 1e-5 --optimizer_name SGD --root_dir ~/cls_data_1_min/ --video_model_name slowfast_r50 --experiment_name final_pipeline_video_audio_caption_wd_drop_lr_sched_vanilla_v4 --vanilla_fusion --batch_size 7 --print_every  10 --spectrogram_model_name resnet18  --device cuda:0 --captions_data_names_pkl_path attention_fusion_default_networks_self_attention_21epochs_caption_modality/ --modalities video audio --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --weight_decay 1e-4 --use_lr_scheduler --dropout 0.2
