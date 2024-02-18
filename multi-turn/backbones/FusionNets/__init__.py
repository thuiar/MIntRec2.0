from .MAG_BERT import MAG_BERT
from .MULT import MULT

multimodal_methods_map = {
    'mag_bert': MAG_BERT,
    'mag_bert_ood': MAG_BERT,
    'mult': MULT,
    'mult_ood': MULT
}