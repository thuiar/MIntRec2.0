#!/usr/bin/bash

for method in 'text_ood'
do
    for text_backbone in 'bert-large-uncased'  
    do
        for ood_detection_method in 'maxlogit'  
        do
            python run.py \
            --dataset 'MIntRec2.0' \
            --data_mode 'ood' \
            --ood_dataset 'MIntRec2.0-OOD' \
            --data_path '/home/sharing/Datasets/MIntRec2.0' \
            --logger_name ${method}_${ood_detection_method} \
            --multimodal_method $method \
            --method ${method}\
            --ood_detection_method $ood_detection_method \
            --ood \
            --train \
            --tune \
            --save_results \
            --save_model \
            --gpu_id '2' \
            --text_backbone $text_backbone \
            --config_file_name $method \
            --results_file_name 'results_text_ood.csv'
        done
    done
done