from .MAG_BERT.manager import MAG_BERT
from .TEXT.manager import TEXT
from .MULT.manager import MULT

method_map = {
    'mag_bert': MAG_BERT,
    'text': TEXT,
    'mult': MULT,
}