class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
        """
        if args.text_backbone.startswith('bert'):
            hyper_parameters = {
                'eval_monitor': ['acc'],
                'train_batch_size': [16],
                'eval_batch_size': 8,
                'test_batch_size': 8,
                'wait_patience': 8,
                'num_train_epochs': [40],
                'multiple_ood': 1,
                'warmup_proportion': 0.1,
                'lr': [2e-5], 
                'weight_decay': 0.1,
                'scale': [1]
            }
        
        return hyper_parameters