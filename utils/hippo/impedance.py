from utils.config import *

def s3_impedance_test():

    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['cifar']),
        flag("model", ['s3']),
        flag("dataset", ['impedance']),
        flag("model.dropout", [0.2]),
        flag("model.n_layers", [2]),
        flag("decoder.mode", ['pool']),
        flag("trainer.max_epochs", [50]),
        flag("optimizer.lr", [0.01]),
        flag("+scheduler.patience", [10]),
        flag("task.torchmetrics", ['[F1,AUROC,Precision,Recall]']),
    ])

    return sweep

def s3_impedance_sweep():

    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['cifar']),
        flag("model", ['s3']),
        flag("dataset", ['impedance']),
        flag("model.dropout", [0.2, 0.3, 0.4]), # higher dropout
        flag("model.n_layers", [1, 2, 4, 6]),
        flag("decoder.mode", ['pool', 'last']), # pool
        flag("trainer.max_epochs", [50]),
        flag("optimizer.lr", [0.01, 4e-3]), # lower lr
        flag("+scheduler.patience", [10]),
        flag("task.torchmetrics", ['[F1,AUROC,Precision,Recall]']),
    ])

    return sweep

def s3_impedance_sweep_2():

    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['cifar']),
        flag("model", ['s3']),
        flag("dataset", ['impedance']),
        flag("model.dropout", [0.5]), # higher dropout
        flag("model.n_layers", [2, 4]),
        flag("decoder.mode", ['pool']), # pool
        flag("trainer.max_epochs", [100]),
        flag("optimizer.lr", [4e-3, 1e-3]), # lower lr
        flag("model.prenorm", [True, False]),
        flag("+scheduler.patience", [10]),
        flag("task.torchmetrics", ['[F1,AUROC,Precision,Recall]']),
    ])

    return sweep

def s3_impedance_sweep_3():

    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ['cifar']),
        flag("model", ['s3']),
        flag("dataset", ['impedance']),
        flag("model.dropout", [0.4]), # higher dropout
        flag("model.n_layers", [4, 6, 8]),
        flag("decoder.mode", ['pool']), # pool
        flag("trainer.max_epochs", [100]),
        flag("optimizer.lr", [4e-3, 1e-3]), # lower lr
        flag("model.prenorm", [True]),
        flag("+scheduler.patience", [10]),
        flag("task.torchmetrics", ['[F1,AUROC,Precision,Recall]']),
    ])

    return sweep


def s4_impedance_global_1():

    sweep = prod([
        flag("train.seed", [0]),
        flag("experiment", ['s4-impedance']),
        flag("model.dropout", [0.3, 0.4, 0.5]),
        flag("optimizer.weight_decay", [0.05, 0.1, 0.20]),
        flag("model.n_layers", [4]),
        flag("model.layer.bidirectional", [True]),
        flag("model.layer.postact", ['glu']),
        lzip([
            flag("optimizer.lr", [0.01, 0.001]),
            flag("+model.layer.lr_dt", [0.01, 0.001]),
        ]),
        
    ])

    return sweep

def s4_impedance_global_2():

    sweep = prod([
        flag("train.seed", [0]),
        flag("experiment", ['s4-impedance']),
        flag("model.dropout", [0.2]),
        flag("optimizer.weight_decay", [0.05, 0.1, 0.20]),
        flag("model.n_layers", [4, 6]),
        flag("model.layer.bidirectional", [True]),
        flag("model.layer.postact", ['glu']),
        lzip([
            flag("optimizer.lr", [0.01]),
            flag("+model.layer.lr_dt", [0.01]),
        ]),
        
    ])

    return sweep

def s4_impedance_global_3():

    sweep = prod([
        flag("train.seed", [0]),
        flag("experiment", ['s4-impedance']),
        flag("model.dropout", [0.3]),
        flag("optimizer.weight_decay", [0.05, 0.1, 0.20]),
        flag("model.n_layers", [4, 6]),
        flag("model.layer.bidirectional", [True]),
        flag("model.layer.postact", ['glu']),
        flag("optimizer.lr", [0.01]),
        flag("+model.layer.lr_dt", [0.01]),
        lzip([
            flag("model.layer.measure", ['fourier', 'fourier_diag', 'hippo', 'random']),
            flag("+model.layer.rank_weight", [1, 1, 1, 0]),    
        ]),
        
    ])

    return sweep

def s4_impedance_final_1():

    sweep = prod([
        flag("train.seed", [1, 2, 3, 4, 5]),
        flag("experiment", ['s4-impedance']),
        flag("model.dropout", [0.3]),
        flag("optimizer.weight_decay", [0.20]),
        flag("model.n_layers", [4, 6]),
        flag("model.layer.bidirectional", [True]),
        flag("model.layer.postact", ['glu']),
        flag("optimizer.lr", [0.01]),
        flag("+model.layer.lr_dt", [0.01]),
        lzip([
            flag("model.layer.measure", ['fourier', 'random']),
            flag("+model.layer.rank_weight", [1, 0]),    
        ]),
        
    ])

    return sweep


# python -m checkpoints.visualize_2 wandb=null experiment=s4-impedance model.dropout=0.3 model.n_layers=4 model.layer.bidirectional=True model.layer.postact=glu train.checkpoint_path=/home/workspace/hippo/outputs/2022-05-07/04-55-32-287949/checkpoints/val/AUROC.ckpt  train.visualizer=impedance loader.drop_last=false