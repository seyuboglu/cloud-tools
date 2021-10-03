from utils.config import *

def sc_repro():

    sweep = [
        flag("train.seed", [0]), 
        flag("+experiment", ['s3-sc'])
    ]
    
    return sweep

def sc_check_resolution():

    sweep = prod([
        flag("train.seed", [0]), 
        flag("+experiment", ['s3-sc']),
        flag("+model.layer.test_resolution", [True]),
        flag("model.layer.cache", [True]),
        flag("optimizer.lr", [1e-3]),
        flag("model.layer.trainable.A", [0]),
        flag("model.layer.trainable.dt", [0]),
    ])
    
    return sweep

def sc_sweep():

    sweep = prod([
        flag("train.seed", [0]),
        flag("+experiment", ['s3-sc']),
        flag("+model.layer.test_resolution", [True]),
        flag("model.layer.cache", [True]),
        flag("optimizer.lr", [1e-2, 4e-3]),
        flag("decoder.mode", ['pool', 'last']),
        flag("model.dropout", [0.15, 0.2]),
    ])

    return sweep

def sc_last_test():

    sweep = prod([
        flag("train.seed", [0]),
        flag("+experiment", ['s3-sc']),
        # flag("+model.layer.test_resolution", [True]),
        # flag("model.layer.cache", [True]),
        flag("optimizer.lr", [2e-3]),
        flag("decoder.mode", ['last']),
        flag("model.dropout", [0.]),
        flag("loader.batch_size", [25]),
        flag("model.layer.l_max", [16000, 'null']),
        flag("model.pool.pool", [4]),
        flag("model.d_model", [256]),
        flag("model.n_layers", [4]),
    ])

    return sweep

def sc_last_test_rerun():

    sweep = prod([
        flag("train.seed", [0]),
        flag("+experiment", ['s3-sc']),
        # flag("+model.layer.test_resolution", [True]),
        # flag("model.layer.cache", [True]),
        flag("optimizer.lr", [1e-3]), # 2e-3 crashes, 1e-3 crashes after a 10 epochs
        flag("decoder.mode", ['last']),
        flag("model.dropout", [0.]),
        flag("loader.batch_size", [25]),
        flag("model.layer.l_max", ['null']),
        flag("model.pool.pool", [4]),
        flag("model.d_model", [256]),
        flag("model.n_layers", [4]),
        flag("loader.eval_resolutions", ['[1,2]']),
    ])

    return sweep

def sc_sweep_2():
    
    sweep = prod([
        flag("train.seed", [0]),
        flag("+experiment", ['s3-sc']),
        # flag("+model.layer.test_resolution", [True]),
        # flag("model.layer.cache", [True]),
        flag("optimizer.lr", [1e-2, 4e-3]),
        flag("decoder.mode", ['pool']),
        flag("model.dropout", [0.15, 0.1]),
        flag("trainer.max_epochs", [150]),
    ])

    return sweep

def sc_sweep_resolution():
    sweep = prod([
        flag("train.seed", [0]),
        flag("+experiment", ['s3-sc']),
        flag("+model.layer.test_resolution", [True]),
        flag("model.layer.cache", [True]),
        flag("model.dropout", [0.1, 0.]),
        flag("model.d_model", [256]),
        flag("model.n_layers", [3]),
        flag("model.layer.l_max", [16000]),
        flag("model.layer.trainable.A", [0]),
        flag("model.layer.trainable.B", [0]),
        flag("model.layer.trainable.dt", [0]),
        flag("model.layer.trainable.C", [0]),
        flag("model.pool.pool", [4]),
        flag("decoder.mode", ['pool', 'last']),
        flag("optimizer.lr", [1e-3, 1e-2]),
        flag("loader.batch_size", [16]),
        flag("trainer.max_epochs", [150]),
    ])

    return sweep

def sc_sweep_resolution_2():
    sweep = prod([
        flag("train.seed", [0]),
        flag("+experiment", ['s3-sc']),
        flag("model.layer.cache", [True]),
        flag("model.dropout", [0.1, 0.15]),
        flag("model.layer.l_max", [16000]),
        flag("decoder.mode", ['pool']),
        flag("loader.batch_size", [16]),
        flag("trainer.max_epochs", [150]),
        flag("optimizer.lr", [2e-3, 4e-3]),
        flag("optimizer.weight_decay", [0.01]),
        # flag("model.d_model", [128]),
        # flag("model.n_layers", [6]),
        # flag("model.layer.trainable.A", [0]),
        # flag("model.layer.trainable.B", [0]),
        # flag("model.layer.trainable.dt", [0]),
        # flag("model.layer.trainable.C", [0]),
        # flag("model.poolk.pool", [1]), # not 4
        # flag("+model.layer.test_resolution", [True]),
        # flag("loader.eval_resolutions", ['[1,2]']),
    ])

    return sweep

def sc_mfcc_sweep():

    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['sc-mfcc']),
        flag("model", ['s3']),
        flag("model.layer.cache", [True]),
        flag("model.dropout", [0.1, 0.2]),
        flag("model.n_layers", [4, 6]),
        flag("model.layer.l_max", [161]),
        flag("decoder.mode", ['pool', 'last']),
        flag("loader.batch_size", [100]),
        flag("trainer.max_epochs", [50]),
        flag("optimizer.lr", [0.01, 4e-3]),
        flag("scheduler.patience", [5]),
    ])

    return sweep

def exprnn_mfcc_sweep():
    
    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['sc-mfcc']),
        flag("model", ['exprnn']),
        flag("model.dropout", [0.]),
        flag("model.n_layers", [1]),
        flag("model.layer.cell.d_model", [256, 512]),
        flag("model.residual", ['N']),
        flag("decoder.mode", ['last']),
        flag("loader.batch_size", [100]),
        flag("trainer.max_epochs", [50]),
        flag("optimizer.lr", [0.001, 0.002, 0.0005]),
    ])

    return sweep

def lipschitzrnn_mfcc_sweep():
    
    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['sc-mfcc']),
        flag("model", ['lipschitzrnn']),
        flag("model.d_model", [256, 512]),
        flag("decoder.mode", ['last']),
        flag("loader.batch_size", [100]),
        flag("trainer.max_epochs", [50]),
        flag("optimizer.lr", [0.001, 0.002, 0.0005]),
    ])

    return sweep

def transformer_mfcc_sweep():
    
    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['sc-mfcc']),
        flag("model", ['transformer']),
        flag("model.dropout", [0., 0.1]),
        flag("model.n_layers", [2, 4]),
        flag("decoder.mode", ['last']),
        flag("loader.batch_size", [16]),
        flag("trainer.max_epochs", [50]),
        flag("optimizer.lr", [0.001]),
    ])

    return sweep

def performer_mfcc_sweep():
    
    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['sc-mfcc']),
        flag("model", ['transformer']),
        flag("model.layer", ['performer']),
        flag("model.dropout", [0., 0.1]),
        flag("model.n_layers", [2, 4]),
        flag("decoder.mode", ['last']),
        flag("loader.batch_size", [16]),
        flag("trainer.max_epochs", [50]),
        flag("optimizer.lr", [0.001]),
    ])

    return sweep


def exprnn_sweep():
    
    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['sc']),
        flag("model", ['exprnn']),
        flag("model.dropout", [0., 0.1]),
        flag("model.n_layers", [2, 4]),
        flag("decoder.mode", ['last']),
        flag("loader.batch_size", [16]),
        flag("trainer.max_epochs", [50]),
        flag("optimizer.lr", [0.001]),
    ])

    return sweep

def lipschitzrnn_sweep():
    
    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['sc']),
        flag("model", ['lipschitzrnn']),
        flag("model.dropout", [0., 0.1]),
        flag("model.n_layers", [2, 4]),
        flag("decoder.mode", ['last']),
        flag("loader.batch_size", [16]),
        flag("trainer.max_epochs", [50]),
        flag("optimizer.lr", [0.001]),
    ])

    return sweep