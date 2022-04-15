from utils.config import *

def scg_trial_sweep():

    sweep = prod([
        flag("train.seed", [0]), 
        flag("pipeline", ['scg']),
        flag("loader.batch_size", [16]),
        flag("model", ['base']),
        flag("model.n_layers", [4, 6]),
        flag("model.d_model", [128]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0.1]),
    ])
    
    return sweep

def scg_trial_sweep_2():

    sweep = prod([
        flag("train.seed", [0]), 
        flag("pipeline", ['scg']),
        flag("loader.batch_size", [16]),
        flag("model", ['base']),
        flag("model.n_layers", [4, 6]),
        flag("model.d_model", [128]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0.]),
        flag("optimizer.lr", [0.01]),
    ])
    
    return sweep

# Now using new version of the SC Generation dataset where 
# the input data = mu law encoding -> mu law decoding -> processed input data
def scg_trial_sweep_3():

    sweep = prod([
        flag("train.seed", [0]), 
        flag("pipeline", ['scg']),
        flag("model", ['base']),
        lzip([
            flag("model.n_layers", [4, 6, 8, 12]),
            flag("loader.batch_size", [24, 16, 16, 12]),
        ]),
        flag("model.d_model", [128]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0.]),
        flag("optimizer.lr", [0.01]),
        flag("model.norm", ['layer']),
        flag("task.metrics", ['bpb']),
    ])
    
    return sweep
