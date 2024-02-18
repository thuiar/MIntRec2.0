from .MAG_BERT.manager import MAG_BERT
from .MAG_BERT.manager_ood import MAG_BERT_OOD
from .TEXT.manager import TEXT
from .TEXT.manager_ood import TEXT_OOD
from .MULT.manager import MULT
from .MULT.manager_ood import MULT_OOD

method_map = {
    'mag_bert': MAG_BERT,
    'text': TEXT,
    'text_ood': TEXT_OOD,
    'mag_bert_ood': MAG_BERT_OOD,
    'mult': MULT,
    'mult_ood': MULT_OOD
}