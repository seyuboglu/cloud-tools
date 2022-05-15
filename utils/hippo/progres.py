from ..config import *


def cifar_progres_test():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:50,bandlimit:null},{resolution:2,epochs:25,bandlimit:null},{resolution:1,epochs:25,bandlimit:null}]"',
                '"[{resolution:4,epochs:50,bandlimit:0.2},{resolution:2,epochs:25,bandlimit:0.4},{resolution:1,epochs:25,bandlimit:null}]"',
                '"[{resolution:4,epochs:50,bandlimit:null},{resolution:2,epochs:40,bandlimit:null},{resolution:1,epochs:10,bandlimit:null}]"',
                '"[{resolution:2,epochs:50,bandlimit:null},{resolution:1,epochs:50,bandlimit:null}]"',
                '"[{resolution:1,epochs:100,bandlimit:null}]"',
            ])
        ]
    )

    return sweep

def cifar_progres_test_2():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:2,epochs:50,bandlimit:null},{resolution:1,epochs:50,bandlimit:null}]"',
                '"[{resolution:2,epochs:100,bandlimit:null}]"',
            ]),
            flag("+loader.img_size", [32]),
        ]
    )

    return sweep

def cifar_progres_test_3(): # this is the same as test_2

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:2,epochs:100,bandlimit:null}]"',
            ]),
            flag("+loader.img_size", [32]),
            flag("model.layer.l_max", [[16, 16]]),
        ]
    )

    return sweep

def cifar_progres_test_4():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:2,epochs:100,bandlimit:null}]"',
            ]),
            flag("+loader.img_size", [32]),
            flag("dataset.augment", [True, False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("loader.train_resolution", [2]),
        ]
    )

    return prod([sweep, [["~callbacks.progressive_resizing"], [""]]])

def cifar_2d_alias_15(): # found bug: linear passthrough encoder
    sweep = prod([
        flag('experiment', [ 's4-cifar-2d', ]),
        flag('train.seed', [1]),
        flag('dataset.augment', [False]),
        flag('model.layer.l_max', [[32, 32]]),
        flag('loader.train_resolution', [2]),
        flag('+loader.img_size', [32]),
        flag('model.d_model', [128]),
        flag('model.layer.d_state', [64]),
        flag('model.n_layers', [4]),
        flag('model.layer.postact', ['glu']),
        flag('model.layer.bidirectional', [True]),
        flag('trainer.max_epochs', [100]),
        flag('scheduler.num_training_steps', [100000]),
        lzip([
            flag('model.dropout', [0.1]),
            flag('optimizer.weight_decay', [0.03]),
        ]),
        flag('model.layer.measure', ['legs', 'fourier']),
        flag('model.layer.trainable', [1]),
        flag('model.layer.dt_max', [1.0]),
        flag('model.layer.dt_min', [0.1]),
        flag('+model.layer.bandlimit', [None]),
    ])
    return sweep

def cifar_progres_test_5():
    # baseline for zero shot test from 16 -> 32 without bandlimiting
    # + 
    # testing 50 + 50 epochs prog res
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:2,epochs:50,bandlimit:null},{resolution:1,epochs:50,bandlimit:null}]"',
                '"[{resolution:2,epochs:100,bandlimit:null}]"',
            ]),
            flag("dataset.augment", [True, False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
        ]
    )

    return sweep

def cifar_progres_baseline_32():
    # standard CIFAR-10 performance, with / without aug
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:1,epochs:100,bandlimit:null}]"',
            ]),
            flag("dataset.augment", [True, False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
        ]
    )

    return sweep

def cifar_progres_zeroshot_16_32():
    # zeroshot test from 16 -> 32
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:2,epochs:100,bandlimit:1.6}]"',
                '"[{resolution:2,epochs:100,bandlimit:0.8}]"',
                '"[{resolution:2,epochs:100,bandlimit:0.4}]"',
                '"[{resolution:2,epochs:100,bandlimit:0.2}]"',
                '"[{resolution:2,epochs:100,bandlimit:0.1}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
        ]
    )
    # TODO: augment True

    return sweep

def cifar_progres_zeroshot_8_16_32():
    # zeroshot test from 8 -> 16, 32
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:100,bandlimit:1.6}]"',
                '"[{resolution:4,epochs:100,bandlimit:0.8}]"',
                '"[{resolution:4,epochs:100,bandlimit:0.4}]"',
                '"[{resolution:4,epochs:100,bandlimit:0.2}]"',
                '"[{resolution:4,epochs:100,bandlimit:0.1}]"',
                '"[{resolution:4,epochs:100,bandlimit:null}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )
    # TODO: augment True

    return sweep

def cifar_progres_zeroshot_8_16_32__2():
    # zeroshot test from 8 -> 16, 32 -- more settings
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:100,bandlimit:0.05}]"',
                '"[{resolution:4,epochs:100,bandlimit:0.025}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )
    # TODO: augment True

    return sweep



def cifar_progres_8_16_32_sweep():
    # progressive resizing from 8 -> 16 -> 32
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                # (50, 25, 25) schedule
                '"[{resolution:4,epochs:50,bandlimit:null},{resolution:2,epochs:25,bandlimit:null},{resolution:1,epochs:25,bandlimit:null}]"',
                '"[{resolution:4,epochs:50,bandlimit:0.1},{resolution:2,epochs:25,bandlimit:0.2},{resolution:1,epochs:25,bandlimit:0.2}]"',
                '"[{resolution:4,epochs:50,bandlimit:0.1},{resolution:2,epochs:25,bandlimit:0.2},{resolution:1,epochs:25,bandlimit:0.4}]"',
                '"[{resolution:4,epochs:50,bandlimit:0.1},{resolution:2,epochs:25,bandlimit:0.2},{resolution:1,epochs:25,bandlimit:null}]"',
                
                # Different epoch schedule: (80, 10, 10)
                '"[{resolution:4,epochs:80,bandlimit:null},{resolution:2,epochs:10,bandlimit:null},{resolution:1,epochs:10,bandlimit:null}]"',
                '"[{resolution:4,epochs:80,bandlimit:0.1},{resolution:2,epochs:10,bandlimit:0.2},{resolution:1,epochs:10,bandlimit:0.2}]"',
                '"[{resolution:4,epochs:80,bandlimit:0.1},{resolution:2,epochs:10,bandlimit:0.2},{resolution:1,epochs:10,bandlimit:0.4}]"',
                '"[{resolution:4,epochs:80,bandlimit:0.1},{resolution:2,epochs:10,bandlimit:0.2},{resolution:1,epochs:10,bandlimit:null}]"',

                # Direct 8 x 8 -> 32 x 32
                '"[{resolution:4,epochs:50,bandlimit:null},{resolution:1,epochs:50,bandlimit:null}]"',
                '"[{resolution:4,epochs:50,bandlimit:0.1},{resolution:1,epochs:50,bandlimit:0.4}]"',
                '"[{resolution:4,epochs:50,bandlimit:0.1},{resolution:1,epochs:50,bandlimit:null}]"',

                '"[{resolution:4,epochs:80,bandlimit:null},{resolution:1,epochs:20,bandlimit:null}]"',
                '"[{resolution:4,epochs:80,bandlimit:0.1},{resolution:1,epochs:20,bandlimit:0.4}]"',
                '"[{resolution:4,epochs:80,bandlimit:0.1},{resolution:1,epochs:20,bandlimit:null}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )
    # TODO: augment True

    return sweep

def cifar_progres_16_32_sweep():
    # progressive resizing from 16 -> 32
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                # Direct 16 x 16 -> 32 x 32
                '"[{resolution:2,epochs:50,bandlimit:null},{resolution:1,epochs:50,bandlimit:null}]"',
                '"[{resolution:2,epochs:50,bandlimit:0.2},{resolution:1,epochs:50,bandlimit:0.4}]"',
                '"[{resolution:2,epochs:50,bandlimit:0.2},{resolution:1,epochs:50,bandlimit:null}]"',

                '"[{resolution:2,epochs:80,bandlimit:null},{resolution:1,epochs:20,bandlimit:null}]"',
                '"[{resolution:2,epochs:80,bandlimit:0.2},{resolution:1,epochs:20,bandlimit:0.4}]"',
                '"[{resolution:2,epochs:80,bandlimit:0.2},{resolution:1,epochs:20,bandlimit:null}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2]]),
        ]
    )
    # TODO: augment True

    return sweep

def cifar_progres_8_32_sweep():
    # progressive resizing from 8 -> 32
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:40,bandlimit:0.1},{resolution:1,epochs:10,bandlimit:0.4},{resolution:4,epochs:40,bandlimit:0.1},{resolution:1,epochs:10,bandlimit:0.4}]"',
                '"[{resolution:4,epochs:40,bandlimit:0.1},{resolution:1,epochs:10,bandlimit:null},{resolution:4,epochs:40,bandlimit:0.1},{resolution:1,epochs:10,bandlimit:null}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )
    # TODO: augment True

    return sweep

def cifar_progres_8_32_sweep_2():
    # progressive resizing from 8 -> 32

    def repeat(schedule, n):
        return ",".join([schedule] * n)

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                f'"[{repeat("{resolution:4,epochs:4,bandlimit:0.1},{resolution:1,epochs:1,bandlimit:0.4}", 20)}]"',
                f'"[{repeat("{resolution:4,epochs:4,bandlimit:0.1},{resolution:1,epochs:1,bandlimit:null}", 20)}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )
    # TODO: augment True

    return sweep

def cifar_progres_8_32_sweep_3():
    # progressive resizing from 8 -> 32

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:80,bandlimit:0.1},{resolution:1,epochs:20,bandlimit:0.4,scheduler:{num_training_steps:20000}}]"',
                '"[{resolution:4,epochs:80,bandlimit:0.1},{resolution:1,epochs:20,bandlimit:null,scheduler:{num_training_steps:20000}}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )
    # TODO: augment True

    return sweep

def cifar_progres_8_32_sweep_4():
    # progressive resizing from 8 -> 32

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:20,bandlimit:0.1},{resolution:1,epochs:20,bandlimit:0.4}]"',
                '"[{resolution:4,epochs:20,bandlimit:0.1},{resolution:1,epochs:20,bandlimit:null}]"',
                '"[{resolution:4,epochs:40,bandlimit:0.1},{resolution:1,epochs:20,bandlimit:0.4}]"',
                '"[{resolution:4,epochs:40,bandlimit:0.1},{resolution:1,epochs:20,bandlimit:null}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )
    # TODO: augment True

    return sweep


def cifar_progres_8_32_sweep_5():
    # progressive resizing from 8 -> 32

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:80,bandlimit:0.1,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,bandlimit:0.4,scheduler:{num_training_steps:20000}}]"',
                '"[{resolution:4,epochs:80,bandlimit:0.1,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,bandlimit:null,scheduler:{num_training_steps:20000}}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )
    # TODO: augment True

    return sweep


def cifar_progres_8_32_sweep_6():
    # progressive resizing from 8 -> 32

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:160,bandlimit:0.1,scheduler:{num_training_steps:160000}},{resolution:1,epochs:40,bandlimit:0.4,scheduler:{num_training_steps:40000}}]"',
                '"[{resolution:4,epochs:160,bandlimit:0.1,scheduler:{num_training_steps:160000}},{resolution:1,epochs:40,bandlimit:null,scheduler:{num_training_steps:40000}}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
            flag("trainer.max_epochs", [200]),
            flag("scheduler.num_training_steps", [200000]),
        ]
    )
    # TODO: augment True

    return sweep



def cifar_progres_16_32_sweep_2():
    # progressive resizing from 16 -> 32

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:2,epochs:80,bandlimit:0.1,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,bandlimit:0.4,scheduler:{num_training_steps:20000}}]"',
                '"[{resolution:2,epochs:80,bandlimit:0.1,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,bandlimit:null,scheduler:{num_training_steps:20000}}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.d_model", [128]),
            flag("model.n_layers", [4]),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2]]),
        ]
    )
    # TODO: augment True

    return sweep

def cifar_progres_final_sweep_1():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:80,bandlimit:0.1,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,bandlimit:null,scheduler:{num_training_steps:20000}}]"',
                '"[{resolution:2,epochs:80,bandlimit:0.2,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,bandlimit:null,scheduler:{num_training_steps:20000}}]"',
            ]),
            flag("dataset.augment", [False]),
            lzip([
                flag("model.n_layers", [6, 8]),
                flag("model.d_model", [256, 512]),    
                flag("optimizer.weight_decay", [0.03, 0.05]),
            ]),
            flag("model.layer.measure", ['fourier', 'legs', 'random-linear', 'random-inv']),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )

    return sweep

def cifar_progres_final_sweep_2():
    # Ablation: what happens if we continue to train with bandlimiting?
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:80,bandlimit:0.1,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,bandlimit:0.5,scheduler:{num_training_steps:20000}}]"',
            ]),
            flag("dataset.augment", [False]),
            lzip([
                flag("model.n_layers", [6]),
                flag("model.d_model", [256]),    
                flag("optimizer.weight_decay", [0.03]),
            ]),
            flag("model.layer.measure", ['fourier', 'legs', 'random-linear', 'random-inv']),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )

    return sweep

def cifar_progres_final_sweep_3():
    # Ablation: training longer on the higher resolution
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:50,bandlimit:0.1,scheduler:{num_training_steps:50000}},{resolution:1,epochs:50,bandlimit:null,scheduler:{num_training_steps:50000}}]"',
                '"[{resolution:4,epochs:50,bandlimit:0.1,scheduler:{num_training_steps:50000}},{resolution:1,epochs:50,bandlimit:0.5,scheduler:{num_training_steps:50000}}]"',
            ]),
            flag("dataset.augment", [False]),
            lzip([
                flag("model.n_layers", [6]),
                flag("model.d_model", [256]),    
                flag("optimizer.weight_decay", [0.03]),
            ]),
            flag("model.layer.measure", ['fourier', 'legs', 'random-linear', 'random-inv']),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )

    return sweep

def cifar_progres_final_sweep_4():
    # Ablation: training with data augmentation
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:80,bandlimit:0.1,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,bandlimit:null,scheduler:{num_training_steps:20000}}]"',
                '"[{resolution:4,epochs:80,bandlimit:0.1,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,bandlimit:0.5,scheduler:{num_training_steps:20000}}]"',
            ]),
            flag("dataset.augment", [True]),
            lzip([
                flag("model.n_layers", [6]),
                flag("model.d_model", [256]),    
                flag("optimizer.weight_decay", [0.03]),
            ]),
            flag("model.layer.measure", ['fourier', 'legs', 'random-linear', 'random-inv']),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )

    return sweep

def cifar_progres_final_sweep_5():
    # Ablation: training without bandlimiting at all
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/s4-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:80,bandlimit:null,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,bandlimit:null,scheduler:{num_training_steps:20000}}]"',
            ]),
            flag("dataset.augment", [False]),
            lzip([
                flag("model.n_layers", [6]),
                flag("model.d_model", [256]),    
                flag("optimizer.weight_decay", [0.03]),
            ]),
            flag("model.layer.measure", ['fourier', 'legs', 'random-linear', 'random-inv']),
            flag("+loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )

    return sweep

def cifar_progres_conv1d_final_sweep_1():
    # Progressive resizing with Conv2D
    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["progres/cnn-cifar-2d"]),
            flag("callbacks.progressive_resizing.stage_params", [
                '"[{resolution:4,epochs:80,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,scheduler:{num_training_steps:20000}}]"',
                '"[{resolution:2,epochs:80,scheduler:{num_training_steps:80000}},{resolution:1,epochs:20,scheduler:{num_training_steps:20000}}]"',
            ]),
            flag("dataset.augment", [False]),
            flag("model.layer.depthwise", [True, False]),
            lzip([
                flag("model.n_layers", [6, 8]),
                flag("model.d_model", [256, 512]),    
                flag("optimizer.weight_decay", [0.03, 0.05]),
            ]),
            flag("loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )

    return sweep

def cifar_progres_conv1d_final_sweep_2():
    # Conv1D results with augmentation; 2 seeds
    sweep = prod(
        [
            flag("train.seed", [0, 1]),
            flag("experiment", ["progres/cnn-cifar-2d"]),
            flag("loader.train_resolution", [1]),
            flag("dataset.augment", [True]),
            flag("model.layer.depthwise", [True, False]),
            lzip([
                flag("model.n_layers", [6, 8]),
                flag("model.d_model", [256, 512]),    
                flag("optimizer.weight_decay", [0.03, 0.05]),
            ]),
            flag("loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2]]),
        ]
    )

    return sweep

def cifar_progres_conv1d_final_sweep_3():
    # Conv1D results lower-res training zero-shot no augmentation; 2 seeds
    sweep = prod(
        [
            flag("train.seed", [0, 1]),
            flag("experiment", ["progres/cnn-cifar-2d"]),
            flag("loader.train_resolution", [4, 2, 1]),
            flag("dataset.augment", [False]),
            flag("model.layer.depthwise", [True, False]),
            lzip([
                flag("model.n_layers", [6, 8]),
                flag("model.d_model", [256, 512]),    
                flag("optimizer.weight_decay", [0.03, 0.05]),
            ]),
            flag("loader.img_size", [32]),
            flag("loader.eval_resolutions", [[1, 2, 4]]),
        ]
    )

    return sweep
