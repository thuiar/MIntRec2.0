#!/usr/bin/bash

for method in 'mag_bert'
do
    for text_backbone in 'bert-large-uncased' 
    do
        for ood_detection_method in  'ma'  
        do
            for video_feats in 'swin-roi'
            do
                for audio_feats in 'wavlm'
                do
                    python run.py \
                    --dataset 'MIntRec2.0' \
                    --ood_dataset 'MIntRec2.0-OOD' \
                    --data_mode 'multiturn' \
                    --data_path '/home/sharing/Datasets/MIntRec2.0' \
                    --logger_name ${method}_${ood_detection_method} \
                    --multimodal_method $method \
                    --multiturn \
                    --method ${method}\
                    --ood_detection_method $ood_detection_method \
                    --train \
                    --ood \
                    --tune \
                    --save_results \
                    --save_model \
                    --gpu_id '2' \
                    --video_feats $video_feats \
                    --audio_feats $audio_feats \
                    --text_backbone $text_backbone \
                    --config_file_name $method \
                    --results_file_name 'results_mag_bert.csv'
                done
            done
        done
    done
done
 