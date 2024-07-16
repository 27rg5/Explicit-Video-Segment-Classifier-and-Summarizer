
echo " "
echo "baseline_default_networks_21epochs"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name baseline_default_networks_21epochs --get_classified_list --vanilla_fusion --eval_dataset_type val


echo " " 
echo "attention_fusion_default_networks_self_attention_21epochs"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_self_attention_21epochs --get_classified_list --eval_dataset_type val


echo " "
echo "attention_fusion_default_networks_self_attention_21epochs_caption_modality"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_self_attention_21epochs_caption_modality --get_classified_list --eval_dataset_type val --mlp_fusion --mlp_object_path '/home/shaunaks/lda_gridsearch_experiments/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl'

echo " "
echo "video_modality"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name video_modality --modalities video --get_classified_list --vanilla_fusion --eval_dataset_type val

echo " "
echo "language_modality"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name language_modality --modalities text --get_classified_list --vanilla_fusion --eval_dataset_type val

echo " "
echo "audio_modality"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name audio_modality --modalities audio --get_classified_list --vanilla_fusion --eval_dataset_type val

echo " "
echo "language_and_video_modality"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name language_and_video_modality --modalities text video --get_classified_list --vanilla_fusion --eval_dataset_type val

echo " "
echo "language_and_audio_modality"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name language_and_audio_modality --modalities text audio --get_classified_list --vanilla_fusion --eval_dataset_type val

echo " "
echo "audio_and_video_modality"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name audio_and_video_modality --modalities video audio --get_classified_list --vanilla_fusion --eval_dataset_type val

echo " " 
echo "attention_fusion_additional_exp_no_caption_increased_params"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_additional_exp_no_caption_increased_params --ablation_for_caption_modality --get_classified_list --eval_dataset_type val
