
# echo " "
# echo "baseline_default_networks_21epochs"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name baseline_default_networks_21epochs --get_classified_list --vanilla_fusion --eval_dataset_type val


# echo " " 
# echo "attention_fusion_default_networks_self_attention_21epochs"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_self_attention_21epochs --get_classified_list --eval_dataset_type val


# echo " "
# echo "attention_fusion_default_networks_self_attention_21epochs_caption_modality"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_self_attention_21epochs_caption_modality --get_classified_list --eval_dataset_type val --mlp_fusion --mlp_object_path '/home/shaunaks/lda_gridsearch_experiments/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl'


# echo "ABLATIONS WITH SELF ATTENTION"
# echo " "
# echo "EVALUATING VIDEO AUDIO TEXT CAPTION" 
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_v1  --get_classified_list --eval_dataset_type val --mlp_fusion --mlp_object_path topic_model_mlps/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl --batch_size 5

# echo " "
# echo "EVALUATING VIDEO AUDIO TEXT" 
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_audio_text  --get_classified_list --eval_dataset_type val  --batch_size 5

# echo " "
# echo "EVALUATING VIDEO TEXT"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_text --modalities video text --get_classified_list --eval_dataset_type val --batch_size 5

# echo " "
# echo "EVALUATING AUDIO TEXT"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_audio_text --modalities text audio --get_classified_list  --eval_dataset_type val --batch_size 5

echo " "
echo "EVALUATING AUDIO VIDEO"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_audio --modalities video audio --get_classified_list --eval_dataset_type val --batch_size 5

# echo " "
# echo "EVALUATING VIDEO"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video --modalities video --get_classified_list --eval_dataset_type val --batch_size 5

# echo " "
# echo "EVALUATING TEXT"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_text --modalities text --get_classified_list --eval_dataset_type val --batch_size 5

# echo " "
# echo "EVALUATING AUDIO"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_audio --modalities audio --get_classified_list --eval_dataset_type val --batch_size 5



# echo " " 
# echo "attention_fusion_additional_exp_no_caption_increased_params"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_additional_exp_no_caption_increased_params --ablation_for_caption_modality --get_classified_list --eval_dataset_type val



echo -e "\n\n ABLATIONS WITH SIMPLE CONCATENATION"

# echo " "
# echo "EVALUATING VIDEO AUDIO TEXT VANILLA" 
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_audio_text_vanilla  --get_classified_list --eval_dataset_type val --vanilla --batch_size 5

# echo " "
# echo "EVALUATING VIDEO TEXT VANILLA"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_text_vanilla --modalities text video --get_classified_list --eval_dataset_type val --vanilla --batch_size 5

# echo " "
# echo "EVALUATING AUDIO TEXT VANILLA"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_audio_text_vanilla --modalities text audio --get_classified_list  --eval_dataset_type val --vanilla --batch_size 5

# echo " "
# echo "EVALUATING AUDIO VIDEO VANILLA"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_audio_vanilla --modalities video audio --get_classified_list --eval_dataset_type val --vanilla --batch_size 5

# echo " "
# echo "EVALUATING VIDEO VANILLA"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_video_vanilla --modalities video --get_classified_list --eval_dataset_type val --vanilla --batch_size 5

# echo " "
# echo "EVALUATING TEXT VANILLA"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_text_vanilla --modalities text --get_classified_list --eval_dataset_type val --vanilla --batch_size 5

# echo " "
# echo "EVALUATING AUDIO VANILLA"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name final_pipeline_audio_vanilla --modalities audio --get_classified_list --eval_dataset_type val --vanilla --batch_size 5


