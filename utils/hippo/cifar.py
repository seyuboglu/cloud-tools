from utils.config import *


def cifar_repro():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-cifar"]),
            flag("optimizer.lr", [0.01, 0.004]),
            flag("loader.batch_size", [32]),
        ]
    )

    return sweep


def cifar_repro_2():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-cifar"]),
            flag("optimizer.lr", [0.01]),
            flag("model.prenorm", [True]),
            flag("loader.batch_size", [32]),
        ]
    )

    return sweep

def cifar_repro_3():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-cifar"]),
            flag("optimizer.lr", [0.01, 0.004]),
            flag("model.prenorm", [True, False]),
            flag('model.layer.bidirectional', [True]),
            flag("loader.batch_size", [32]),
        ]
    )

    return sweep