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