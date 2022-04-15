from utils.config import flag, chain, prod, lzip

def vc_s3_sweep():
    sweep = prod([
        flag("pipeline", ['sc']),
        flag("dataset", ['voxceleb']),
        flag("model", ['s3']),
        flag("model.d_model", [64]),
        flag("model.layer.d_model", [64]),
        flag("model.pool.pool", [4]),
        flag("model.pool.expand", [2]),
        flag("model.n_layers", [4, 6]),
        flag("loader.batch_size", [64]),
        flag("loader.num_workers", [16]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0.]),
        flag("optimizer.lr", [1e-2, 4e-3, 1e-3, 5e-4]),
        flag("model.norm", ['batch']),
    ])
      
    return sweep

def vc_s3_sweep_2():
    sweep = prod([
        flag("pipeline", ['sc']),
        flag("dataset", ['voxceleb']),
        flag("dataset.clip_length", [3, 1]),
        flag("model", ['s3']),
        flag("model.d_model", [64]),
        flag("model.layer.d_state", [64]),
        flag("model.pool.pool", [4]),
        flag("model.pool.expand", [2]),
        flag("model.n_layers", [4, 2]),
        flag("loader.batch_size", [64]),
        flag("loader.num_workers", [16]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0., 0.2]),
        flag("optimizer.lr", [4e-3]),
        flag("model.norm", ['layer']),
    ])
      
    return sweep

def vc_s3_sweep_3():
    sweep = prod([
        flag("pipeline", ['sc']),
        flag("dataset", ['voxceleb']),
        flag("dataset.clip_length", [3]),
        flag("model", ['s3']),
        flag("model.d_model", [64]),
        flag("model.layer.d_state", [64]),
        # flag("model.pool.pool", [4]),
        # flag("model.pool.expand", [2]),
        flag("model.n_layers", [4]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0.]),
        flag("+encoder._name_", ['conv1d']),
        lzip([
            flag("+encoder.kernel_size", [400, 8]), # 25ms, 0.5ms
            flag("+encoder.stride", [160, 8]), # 10ms, 0.5ms
            flag("+encoder.padding", [160, 0]),
            flag("loader.batch_size", [128, 64]),
        ]),
        flag("optimizer.lr", [4e-3]),
        flag("model.norm", ['layer', 'batch']),
    ])
      
    return sweep

def vc_s3_sweep_4():
    sweep = prod([
        flag("pipeline", ['sc']),
        flag("dataset", ['voxceleb']),
        flag("dataset.clip_length", [3]),
        flag("model", ['s3']),
        flag("model.d_model", [128]),
        flag("model.layer.d_state", [64]),
        # flag("model.pool.pool", [4]),
        # flag("model.pool.expand", [2]),
        flag("model.n_layers", [8]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0.1]),
        flag("+encoder._name_", ['conv1d']),
        lzip([
            flag("+encoder.kernel_size", [8]),
            flag("+encoder.stride", [8]),
            flag("+encoder.padding", [0]),
            flag("loader.batch_size", [32]),
        ]),
        flag("optimizer.lr", [4e-3]),
        flag("model.norm", ['layer', 'batch']),
    ])
      
    return sweep


def vc_s3_sweep_aug_1(): # too slow to train: need to remove dropout
    sweep = prod([
        flag("pipeline", ['sc']),
        flag("dataset", ['voxceleb']),
        flag("dataset.clip_length", [3]),
        flag("dataset.noise", [True]),
        flag("dataset.effects", [True]),
        flag("model", ['s3']),
        flag("model.d_model", [64]),
        flag("model.layer.d_state", [64]),
        # flag("model.pool.pool", [4]),
        # flag("model.pool.expand", [2]),
        flag("model.n_layers", [6]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0.1]),
        flag("+encoder._name_", ['conv1d']),
        lzip([
            flag("+encoder.kernel_size", [8]),
            flag("+encoder.stride", [8]),
            flag("+encoder.padding", [0]),
            flag("loader.batch_size", [64]),
        ]),
        flag("optimizer.lr", [4e-3]),
        flag("model.norm", ['layer', 'batch']),
    ])
      
    return sweep


def vc_s3_layernorm_8gpu():
    sweep = prod([
        flag("pipeline", ['sc']),
        flag("dataset", ['voxceleb']),
        flag("model", ['s3']),
        flag("model.d_model", [64]),
        flag("model.layer.d_model", [64]),
        flag("model.pool.pool", [4]),
        flag("model.pool.expand", [2]),
        flag("model.n_layers", [4]),
        flag("loader.batch_size", [64]),
        flag("loader.num_workers", [64]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0.]),
        flag("optimizer.lr", [4e-3]),
        flag("model.norm", ['layer']),
        flag("trainer.gpus", [8]),
    ])
      
    return sweep



def vc_s3_sweep_5():
    sweep = prod([
        flag("pipeline", ['sc']),
        flag("dataset", ['voxceleb']),
        flag("dataset.clip_length", [3]),
        flag("dataset.noise", [False, True]),
        flag("model", ['s3']),
        flag("model.d_model", [32]),
        flag("model.layer.d_state", [32]),
        # flag("model.pool.pool", [4]),
        # flag("model.pool.expand", [2]),
        flag("model.n_layers", [12]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0.1]),
        flag("+encoder._name_", ['conv1d']),
        lzip([
            flag("+encoder.kernel_size", [8]),
            flag("+encoder.stride", [8]),
            flag("+encoder.padding", [0]),
            flag("loader.batch_size", [32]),
        ]),
        flag("optimizer.lr", [1e-2]),
        flag("optimizer.weight_decay", [1e-2]),
        flag("model.norm", ['layer', 'batch']),
    ])
      
    return sweep


def vc_s3_sweep_vsmall_1():
    sweep = prod([
        flag("pipeline", ['sc']),
        flag("dataset", ['voxceleb']),
        flag("dataset.clip_length", [3]),
        flag("dataset.num_classes", [10]),
        flag("dataset.noise", [False, True]),
        flag("dataset.self_normalize", [False, True]),
        flag("model", ['base']),
        flag("model.d_model", [32]),
        flag("model.layer.d_state", [32]),
        flag("model.pool.pool", [4]),
        flag("model.pool.expand", [2]),
        flag("model.n_layers", [4]),
        flag("model.prenorm", [True]),
        flag("model.dropout", [0.0]),
        # lzip([
        #     flag("+encoder._name_", ['conv1d']),
        #     flag("+encoder.kernel_size", [8]),
        #     flag("+encoder.stride", [8]),
        #     flag("+encoder.padding", [0]),
        #     flag("loader.batch_size", [32]),
        # ]),
        flag("optimizer.lr", [4e-3]),
        flag("optimizer.weight_decay", [0.]),
        flag("model.norm", ['layer', 'batch']),
        flag("loader.batch_size", [16]),
        flag("decoder.mode", ["pool", "last"]),
    ])
      
    return sweep
