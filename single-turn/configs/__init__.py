
pretrained_models_path = {
    'bert-base-uncased': '/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/',
    'bert-large-uncased':'/home/sharing/disk1/pretrained_embedding/bert/bert-large-uncased',
    'roberta-base': '/home/sharing/disk1/pretrained_embedding/roberta/roberta-base',
    'roberta-large':'/home/sharing/disk1/pretrained_embedding/roberta/roberta-large',
    't5-base': '/home/sharing/disk1/pretrained_embedding/t5/t5-base',
    't5-large': '/home/sharing/disk1/pretrained_embedding/t5/t5-large',
    'xclip-base-patch32': '/home/sharing/disk1/pretrained_embedding/xclip/xclip-base-patch32-8-frames'
}

video_feats_path = {
    'resnet50-roi': 'resnet50_roi_feats.pkl',
    'alphapose-keypoints': 'keypoints_feats.pkl',
    'xclip-pixels': 'xclip_feats.pkl',
    'swin-full': 'swin_full_feats.pkl',
    'swin-keyframes': 'swin_keyframes.pkl',
    'swin-roi': 'swin_roi_binary.pkl',
    'swin-mask': 'swin_mask_feats.pkl',
    'alphapose-keypoints-body': 'keypoints_body.pkl',
    'alphapose-keypoints-face': 'keypoints_face.pkl',
    'xclip-8-pixels': 'xclip_8_feats.pkl',
    'video-swin-mask': 'video_swin_mask.pkl',
    'swin-mask-roi': 'swin_mask_roi.pkl',
    'normalized-alphapose-keypoints': 'normalized_keypoints_feats.pkl',
    'normalized-alphapose-face': 'normalized_keypoints_face.pkl',
    'aligned-swin-roi': 'align_bert_large_swin-roi_video.pkl'
}

audio_feats_path = {
    'wav2vec2': 'wav2vec2_feats.pkl',
    'wavlm': 'wavlm_feats_binary.pkl',
    'aligned-wavlm': 'align_bert_large_wavml_audio.pkl'
}