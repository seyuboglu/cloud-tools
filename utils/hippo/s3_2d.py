from utils.config import *

def s3_2d_cifar_sweep_1():

    sweep = prod([
        flag("train.seed", [0]), 
        flag("pipeline", ['cifar']),
        flag("model", ['s3_2d']),
        flag("model.dropout", [0.2, 0.3]),
        flag("optimizer.lr", [1e-2, 4e-3]),
        flag("model.d_model", [128]),
        flag("model.n_layers", [4, 6]),
        flag("model.prenorm", [True]),
        flag("model.layer.tie", [False, True]),
    ])
    
    return sweep