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
        ood_detection_parameters = {
            'sbm':{
                'temperature': [1e4],
                'scale': [20]
            },
            'hub':{
                'temperature': [1e6],
                'scale': [20],
                'k': [10],
                'alpha': [1.0]
            }
        }
        if args.text_backbone.startswith('bert'):
            hyper_parameters = {
                'eval_monitor': ['acc'],
                'train_batch_size': 16,
                'eval_batch_size': 8,
                'test_batch_size': 8,
                'wait_patience': 8,
                'num_train_epochs': [40],
                'multiple_ood': 1,
                'warmup_proportion': 0.1,
                'lr': [0.00001], 
                'weight_decay': 0.1,
                'scale': [1]

            }
        elif args.text_backbone.startswith('roberta'):
            hyper_parameters = {
                'eval_monitor': ['acc'],
                'train_batch_size': 16,
                'eval_batch_size': 8,
                'test_batch_size': 8,
                'wait_patience': 8,
                'num_train_epochs': [40],
                'warmup_proportion': 0.1,
                'lr':  [0.00001], 
                'weight_decay': 0.1,
            }
        
        if args.ood_detection_method in ood_detection_parameters.keys():   
            ood_parameters = ood_detection_parameters[args.ood_detection_method]
            hyper_parameters.update(ood_parameters)
        
        return hyper_parameters