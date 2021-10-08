from utils.config import *


def mnist_transformer():
    # python -m train

    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ["mnist"]),
        flag("model", ["transformer"]),
        flag("model.layer.0.n_heads", [8]),
        flag("model.d_model", [128]),
        flag("model.layer.0.causal", [False]),
        flag("model.n_layers", [2]),
        flag("decoder.mode", ['pool']),
        flag("model.prenorm", [True]),
        flag("optimizer.lr", [0.001, 0.0005]),
        flag("loader.batch_size", [50]),
    ])

    return sweep

def mnist_performer():

    sweep = prod([
        flag("train.seed", [0]),
        flag("pipeline", ["mnist"]),
        flag("model", ["transformer"]),
        flag("+model/layer", ["performer"]),
        flag("model.d_model", [128]),
        flag("model.layer.0.causal", [False]),
        flag("model.n_layers", [2]),
        flag("decoder.mode", ['pool']),
        flag("model.prenorm", [True]),
        flag("optimizer.lr", [0.001, 0.0005]),
        flag("loader.batch_size", [50]),
    ])

    return sweep