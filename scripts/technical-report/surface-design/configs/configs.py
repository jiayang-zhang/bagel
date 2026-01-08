from yacs.config import CfgNode as CN

_C = CN()


_C.SETUP = CN()
_C.SETUP.USE_MODAL = False

# Move probabilities
_C.MOVE = CN()
_C.MOVE.SUB = None
_C.MOVE.ADD = None
_C.MOVE.REMOVE = None
_C.MOVE.SWAP = None

# Optimization params
_C.OPTIM = CN()
_C.OPTIM.HIGH_TEMP = None
_C.OPTIM.LOW_TEMP = None
_C.OPTIM.N_STEPS_HIGH = None
_C.OPTIM.N_STEPS_LOW = None
_C.OPTIM.N_CYCLES = None

def get_cfg_defaults():
    return _C.clone()