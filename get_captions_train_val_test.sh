
dataset_types=("train" "val" "test")
for dataset_type in "${dataset_types[@]}"
do
    echo -e "\n Extracting captions for ${dataset_type}"
    python -W ignore eval.py --root_dir_path ~/cls_data_1_min --experiment_name attention_fusion_default_networks_self_attention_21epochs_caption_modality --get_classified_list --eval_dataset_type "$dataset_type" --run_caption_model --batch_size 10
done