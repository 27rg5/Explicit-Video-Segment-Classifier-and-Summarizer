
# echo "baseline_default_networks_100epochs"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name baseline_default_networks_100epochs --get_classified_list --vanilla_fusion
echo "attention_fusion_default_networks_pairwise_21epochs"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_pairwise_21epochs --get_classified_list --pairwise_attention_modalities --eval_dataset_type val
# echo "\nattention_fusion_default_networks_self_attention_21epochs"
# python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_self_attention_21epochs --get_classified_list
echo " "
echo "attention_fusion_default_networks_pairwise_21epochs_mlp_fusion_concat"
python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_pairwise_21epochs_mlp_fusion_concat --get_classified_list --pairwise_attention_modalities --eval_dataset_type val

