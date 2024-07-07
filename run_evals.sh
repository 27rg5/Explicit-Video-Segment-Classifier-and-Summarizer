
# echo " "
# echo "baseline_default_networks_100epochs"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name baseline_default_networks_100epochs --get_classified_list --vanilla_fusion --eval_dataset_type val

# echo " "
# echo "attention_fusion_default_networks_pairwise_21epochs"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_pairwise_21epochs --get_classified_list --pairwise_attention_modalities --eval_dataset_type val

# echo " " 
# echo "attention_fusion_default_networks_self_attention_21epochs"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_self_attention_21epochs --get_classified_list --eval_dataset_type val

echo " "
echo "attention_fusion_default_networks_pairwise_21epochs_mlp_fusion_concat"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_pairwise_21epochs_mlp_fusion_concat --get_classified_list --pairwise_attention_modalities --eval_dataset_type val --mlp_fusion --mlp_object_path '/home/shaunaks/lda_gridsearch_experiments/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl'

echo " "
echo "attention_fusion_default_networks_pairwise_21epochs_mlp_fusion_concat_weighted_loss_for_mlp_only"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_pairwise_21epochs_mlp_fusion_concat_weighted_loss_for_mlp_only --get_classified_list --pairwise_attention_modalities --eval_dataset_type val --mlp_fusion --mlp_object_path '/home/shaunaks/lda_gridsearch_experiments/bertopic/best_model_bertopic_test_f1_0.782608695652174.pkl' --weighted_loss_mlp_fusion