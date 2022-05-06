from ..config import *

def convnext_sweep_1():

    sweep = prod([
        flag("experiment", ["convnext-celeba-all"]),
        flag("loader.batch_size", [1024]),
        flag("dataset.res", [[128, 128], [64, 64]]),
    ])

    return sweep

def convnext_s4nd_sweep_1():

    sweep = prod([
        flag("experiment", ["convnext-s4nd-celeba-all"]),
        flag("loader.batch_size", [1024]),
        flag("dataset.res", [[128, 128], [64, 64]]),
    ])

    return sweep

def convnext_sweep_2():

    sweep = prod([
        flag("experiment", ["convnext-celeba-all"]),
        flag("loader.batch_size", [1024]),
        flag("dataset.res", [[128, 128], [64, 64]]),
    ])

    return sweep

def convnext_s4nd_sweep_2():

    sweep = prod([
        flag("experiment", ["convnext-s4nd-celeba-all"]),
        flag("loader.batch_size", [1024]),
        flag("dataset.res", [[128, 128], [64, 64]]),
    ])

    return sweep

def convnext_sweep_3():

    sweep = prod([
        flag("experiment", ["convnext-celeba-all"]),
        flag("loader.batch_size", [1024]),
        flag("dataset.res", [[128, 128], [64, 64]]),
        flag("optimizer.weight_decay", [0.05, 0.10]),
    ])

    return sweep

def convnext_s4nd_sweep_3():

    sweep = prod([
        flag("experiment", ["convnext-s4nd-celeba-all"]),
        flag("loader.batch_size", [1024]),
        flag("dataset.res", [[128, 128], [64, 64]]),
        flag("optimizer.weight_decay", [0.05, 0.10]),
    ])

    return sweep

def convnext_s4nd_sweep_4():

    sweep = prod([
        flag("experiment", ["convnext-s4nd-celeba-all"]),
        flag("loader.batch_size", [1024]),
        flag("dataset.res", [[128, 128], [64, 64]]),
        flag("optimizer.weight_decay", [0.10]),
        flag("model.layer.dropout", [0.1, 0.2]),
    ])

    return sweep

def convnext_s4nd_sweep_5():

    sweep = prod([
        flag("experiment", ["convnext-s4nd-celeba-all"]),
        flag("loader.batch_size", [1024]),
        flag("dataset.res", [[128, 128]]),
        flag("optimizer.weight_decay", [0.20, 0.50, 1.0, 5.0]),
    ])

    return sweep

def convnext_sweep_4():

    sweep = prod([
        flag("experiment", ["convnext-celeba-all"]),
        flag("loader.batch_size", [1024]),
        flag("dataset.res", [[128, 128]]),
        flag("optimizer.weight_decay", [0.05, 0.10, 0.20, 0.50]),
    ])

    return sweep

def convnext_s4nd_sweep_6():
    # Weight decay sweep: 1.0 does best, 2.0 / 5.0 are too high
    sweep = prod([
        flag("experiment", ["convnext-s4nd-celeba-all"]),
        flag("loader.batch_size", [1024]),
        flag("dataset.res", [[128, 128]]),
        flag("optimizer.weight_decay", [0.20, 0.50, 1.0, 2.0, 5.0]),
    ])

    return sweep

def convnext_s4nd_new_sweep_1():
    sweep = prod([
        flag("experiment", ["convnext-s4nd-celeba-all"]),
        flag("loader.batch_size", [512]),
        flag("dataset.res", [[128, 128]]),
        # Train on 128 x 128 images
        flag("loader.train_resolution", [1.25]),
        # Test on 64 x 64, 128 x 128 images, 160 x 160 images
        flag("loader.eval_resolutions", [[0.5, 1, 1.25]]),
        flag("+loader.img_size", [160]),
        flag("+loader.channels_last", [False]),
        # Manually set the kernel sizes to 1.25x what would be set originally
        flag("model.stem_l_max", [[20, 20]]),
        flag("model.img_size", [[160, 160]]),
        flag("optimizer.weight_decay", [1.0]),
        lzip(
            [
                flag(
                    "model.layer.measure",
                    [
                        "fourier_diag",
                        "random",
                    ],
                ),
                flag("+model.layer.rank_weight", [1, 0]),
            ]
        ),
        flag("+model.layer.bandlimit", [0.2, 0.4, 0.8, 1.6, None]),
    ])

    return sweep

def convnext_s4nd_new_sweep_2():
    sweep = prod([
        flag("experiment", ["convnext-s4nd-celeba-all"]),
        flag("loader.batch_size", [512]),
        flag("dataset.res", [[128, 128]]),
        # Train on 128 x 128 images
        flag("loader.train_resolution", [1.25]),
        # Test on 64 x 64, 128 x 128 images, 160 x 160 images
        flag("loader.eval_resolutions", [[1, 1.25, 2.50]]),
        flag("+loader.img_size", [160]),
        flag("+loader.channels_last", [False]),
        # Manually set the kernel sizes to 1.25x what would be set originally
        flag("model.stem_l_max", [[20, 20]]),
        flag("model.img_size", [[160, 160]]),
        flag("optimizer.weight_decay", [1.0]),
        lzip(
            [
                flag(
                    "model.layer.measure",
                    [
                        "fourier",
                        "legs",
                    ],
                ),
                flag("+model.layer.rank_weight", [1, 1]),
            ]
        ),
        flag("+model.layer.bandlimit", [0.1, 0.2, 0.4, 0.8, 1.6, None]),
    ])

    return sweep
