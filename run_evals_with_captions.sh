
echo "ABLATIONS WITH SELF ATTENTION"
echo " "
echo "EVALUATING VIDEO TEXT CAPTION"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_text_caption --modalities video text --get_classified_list --eval_dataset_type val --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

echo " "
echo "EVALUATING AUDIO TEXT CAPTION"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_audio_text_caption --modalities text audio --get_classified_list  --eval_dataset_type val --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

echo " "
echo "EVALUATING AUDIO VIDEO CAPTION"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_audio_caption --modalities video audio --get_classified_list --eval_dataset_type val --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

echo " "
echo "EVALUATING VIDEO CAPTION"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_caption --modalities video --get_classified_list --eval_dataset_type val --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

echo " "
echo "EVALUATING TEXT CAPTION"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_text_caption --modalities text --get_classified_list --eval_dataset_type val --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

echo " "
echo "EVALUATING AUDIO CAPTION"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_audio_caption --modalities audio --get_classified_list --eval_dataset_type val --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl



# # echo " " 
# # echo "attention_fusion_additional_exp_no_caption_increased_params"
# # python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_additional_exp_no_caption_increased_params --ablation_for_caption_modality --get_classified_list --eval_dataset_type val



echo -e "\n\n ABLATIONS WITH SIMPLE CONCATENATION"
echo " "
echo "EVALUATING VIDEO TEXT CAPTION VANILLA"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_text_caption_vanilla --modalities text video --get_classified_list --eval_dataset_type val --vanilla --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

echo " "
echo "EVALUATING AUDIO TEXT CAPTION VANILLA"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_audio_text_caption_vanilla --modalities text audio --get_classified_list  --eval_dataset_type val --vanilla --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

echo " "
echo "EVALUATING AUDIO VIDEO CAPTION VANILLA"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_audio_caption_vanilla --modalities video audio --get_classified_list --eval_dataset_type val --vanilla --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

echo " "
echo "EVALUATING VIDEO CAPTION VANILLA"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_caption_vanilla --modalities video --get_classified_list --eval_dataset_type val --vanilla --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

echo " "
echo "EVALUATING TEXT CAPTION VANILLA"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_text_caption_vanilla --modalities text --get_classified_list --eval_dataset_type val --vanilla --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

echo " "
echo "EVALUATING AUDIO CAPTION VANILLA"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_audio_caption_vanilla --modalities audio --get_classified_list --eval_dataset_type val --vanilla --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl






################################################### REGULARIZATION EXPERIMENTS #############################################
# echo " "
# echo "EVALUATING AUDIO VIDEO CAPTION WD 1E-4 LR 1E-4 DROP 0.3 VANILLA"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_audio_caption_wd_drop_lr_sched_vanilla --modalities video audio --dropout 0.3 --get_classified_list --eval_dataset_type val --vanilla --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

# echo " "
# echo "EVALUATING AUDIO VIDEO CAPTION WD 1E-4 LR 2E-4 DROP 0.2 VANILLA"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_audio_caption_wd_drop_lr_sched_vanilla_v2 --modalities video audio --dropout 0.2 --get_classified_list --eval_dataset_type val --vanilla --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

# echo " "
# echo "EVALUATING AUDIO VIDEO CAPTION WD 1E-4 LR 2E-4 DROP 0.2 DATA AUGMENTATION VANILLA"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_audio_caption_wd_drop_lr_sched_vanilla_v3 --modalities video audio --dropout 0.2 --get_classified_list --eval_dataset_type val --vanilla --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl

# echo " "
# echo "EVALUATING AUDIO VIDEO CAPTION WD 1E-4 LR 2E-4 DROP 0.2 DATA AUGMENTATION ONECYCLE VANILLA"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_audio_caption_wd_drop_lr_sched_vanilla_v4 --modalities video audio --dropout 0.2 --get_classified_list --eval_dataset_type val --vanilla --batch_size 5 --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl


