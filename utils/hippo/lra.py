from utils.config import *

def pathfinder_repro():

    sweep = prod([
        flag("train.seed", [0]), 
        flag("+experiment", ['s3-lra-pathfinder']),
        flag("decoder.mode", ['last', 'pool']),
    ])
    
    return sweep
# pool never learns anything ^, last repros

def pathfinder_pool():

    sweep = prod([
        flag("train.seed", [0]), 
        flag("+experiment", ['s3-lra-pathfinder']),
        flag("decoder.mode", ['pool']),
        flag("model.layer.lr.dt", [5e-4]),
        flag("optimizer.lr", [5e-4]),
        flag("model.dropout", [0., 0.1]),
    ])
    
    return sweep


def imdb_repro():

    sweep = prod([
        flag("train.seed", [0]), 
        flag("+experiment", ['s3-lra-imdb']),
        flag("decoder.mode", ['last', 'pool']),
        flag("model.prenorm", [True, False]),
        flag("model.n_layers", [4]), # sweep 6 later
        flag("model.d_model", [64]),
        flag("model.dropout", [0.15]),
        flag("model.norm", ['layer', 'batch']),
        flag("optimizer.lr", [0.002]),
        flag("optimizer.weight_decay", [0., 0.01]),
    ])
    
    return sweep

def aan_repro():

    sweep = prod([
        flag("train.seed", [0]), 
        flag("+experiment", ['s3-lra-aan']),
        flag("decoder.mode", ['last', 'pool']),
        flag("model.prenorm", [True]),
        flag("model.n_layers", [4]), # sweep 6 later
        flag("model.d_model", [256]),
        flag("model.dropout", [0., 0.1]),
        flag("model.norm", ['layer', 'batch']),
        flag("optimizer.lr", [0.01]),
    ])
    
    return sweep

## NeurIPS 2022

def pathx_sweep_1():

    sweep = prod([
        flag("experiment", ['s4-lra-pathx-test']),
        flag("model.layer.bidirectional", [True, False]),
        lzip([
            flag("model.layer.dt_min", [1e-3, 1e-4]),
            flag("model.layer.dt_max", [1e-1, 1e-2]),
        ]),
        flag("model.layer.measure", ['hippo', 'legs']),
    ])

    return sweep

def aan_sweep_1():

    sweep = prod([
        flag("experiment", ['s4-lra-pathx-test']),
        flag("model.layer.bidirectional", [True, False]),
        lzip([
            flag("model.layer.dt_min", [1e-3, 1e-4]),
            flag("model.layer.dt_max", [1e-1, 1e-2]),
        ]),
        flag("model.layer.measure", ['hippo', 'legs']),
    ])

    return sweep