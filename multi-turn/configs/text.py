class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
        """
        if args.text_backbone.startswith('bert'):
            hyper_parameters = {
                'eval_monitor': ['acc'],
                'train_batch_size': [2],
                'eval_batch_size': [2],
                'test_batch_size': [2],
                'wait_patience': [3],
                'num_train_epochs': [40],
                'multiple_ood': 1,
                'warmup_proportion': [0.1],
                'lr': [1e-5], 
                'use_scheduler': [False], 
                'weight_decay': 0.1,
                'scale': [1],
                'train_bs': [16],
            }
        
        return hyper_parameters