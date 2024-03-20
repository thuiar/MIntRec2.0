from .FeatureNets import BERTEncoder, RoBERTaEncoder

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder,
                    'bert-large-uncased': BERTEncoder,
                    'roberta-base': RoBERTaEncoder,
                    'roberta-large': RoBERTaEncoder
                }