#!/usr/bin/bash
 
for method in 'text_ood'
do
    for text_backbone in 'bert-large-uncased'  
    do
        for ood_detection_method in 'maxlogit'  
        do
            python run_single_turn.py \
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

# for seed in 0 1 2 3 4 5 6 7 8 9
# do
#     for method in 'text'
#     do
#         for text_backbone in 'bert-large-uncased' 'roberta-base' 'bert-large-uncased' 'roberta-large' # 'bert-large-uncased'
#         do
#             for ood_detection_method in 'residual' 'maxlogit' 'ma' 'energy' 'vim' 
#             do
#                 python run.py \
#                 --dataset 'MIntRec2.0' \
#                 --ood_dataset 'MIntRec2.0-OOD' \
#                 --data_path '/home/sharing/Datasets/MIntRec2.0' \
#                 --logger_name ${method}_${ood_detection_method} \
#                 --multimodal_method $method \
#                 --method ${method}\
#                 --ood_detection_method $ood_detection_method \
#                 --ood \
#                 --tune \
#                 --save_results \
#                 --save_model \
#                 --seed $seed \
#                 --gpu_id '1' \
#                 --video_feats_path 'video_feats.pkl' \
#                 --audio_feats_path 'audio_feats.pkl' \
#                 --text_backbone $text_backbone \
#                 --config_file_name $method \
#                 --results_file_name 'results_mintrec_text.csv'
#             done
#         done
#     done
# done
 