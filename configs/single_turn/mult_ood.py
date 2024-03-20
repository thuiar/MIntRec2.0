class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            dst_feature_dims (int): The destination dimensions (assume d(l) = d(v) = d(t)).
            nheads (int): The number of heads for the transformer network.
            n_levels (int): The number of layers in the network.
            attn_dropout (float): The attention dropout.
            attn_dropout_v (float): The attention dropout for the video modality.
            attn_dropout_a (float): The attention dropout for the audio modality.
            relu_dropout (float): The relu dropout.
            embed_dropout (float): The embedding dropout.
            res_dropout (float): The residual block dropout.
            output_dropout (float): The output layer dropout.
            text_dropout (float): The dropout for text features.
            grad_clip (float): The gradient clip value.
            attn_mask (bool): Whether to use attention mask for Transformer. 
            conv1d_kernel_size_l (int): The kernel size for temporal convolutional layers (text modality).  
            conv1d_kernel_size_v (int):  The kernel size for temporal convolutional layers (video modality).  
            conv1d_kernel_size_a (int):  The kernel size for temporal convolutional layers (audio modality).  
            lr (float): The learning rate of backbone.
        """

        ood_detection_parameters = {
            'sbm':{
                'temperature': [1e6],
                'scale': [20]
            },
            'hub':{
                'temperature': [1e6],
                'scale': [20],
                'k': [50],
                'alpha': [1.0]
            }
        }

        hyper_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': False,
            'eval_monitor': ['acc'],
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8,
            'num_train_epochs': [40],
            'dst_feature_dims': [80],
            'nheads': [4],
            'n_levels': [8],
            'attn_dropout': 0.0,
            'attn_dropout_v': [0.1],
            'attn_dropout_a': [0.1],
            'relu_dropout': 0.3,
            'embed_dropout': 0.0,
            'res_dropout': [0],
            'output_dropout': [0],
            'text_dropout': [0],
            'grad_clip': 0.5, 
            'attn_mask': True,
            'conv1d_kernel_size_l': 5,
            'conv1d_kernel_size_v': 1,
            'conv1d_kernel_size_a': 1,
            'lr': [5e-6],
            'scale': [1]
        }
        return hyper_parameters