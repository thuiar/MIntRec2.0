from .energy import func as ENERGY
from .ma import func as MA
from .vim import func as VIM
from .maxlogit import func as MAXLOGIT
from .msp import func as MSP
from .residual import func as RESIDUAL

ood_detection_map = {
    'energy': ENERGY,
    'ma': MA,
    'vim': VIM,
    'maxlogit': MAXLOGIT,
    'msp': MSP,
    'residual': RESIDUAL
}