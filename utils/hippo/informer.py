from utils.config import *

"""
--data ETTh1 
### M
48,48,24
96,48,48
168,168,168
168,168,336
336,336,720

### S
720,168,24
720,168,48
720,336,168
720,336,336
720,336,720
"""

def etth1_sweep_M(): # launched | relaunched

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("+experiment", ["s3-informer-etth"]),
            flag("decoder.mode", ["last"]), # "pool" is terrible
            flag(
                "dataset.size",
                [
                    "[336,336,720]",
                    "[168,168,336]",
                    "[168,168,168]",
                    "[96,48,48]",
                    "[48,48,24]",
                ],
            ),
            flag("dataset.timeenc", [0]), # 1 doesn't help
            flag("dataset.variant", [0]),
            flag("dataset.features", ["M"]),
            flag("model.n_layers", [1]), # 2 layers don't help
            flag("model.dropout", [0.25, 0.3]), # high dropout helps 
            flag("optimizer.lr", [0.01, 0.004]), # adding lr sweep
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

def etth1_sweep_S(): # launched # relaunched

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("+experiment", ["s3-informer-etth"]),
            flag("decoder.mode", ["last"]), # "pool" is terrible
            flag(
                "dataset.size",
                [
                    "[720,336,720]",
                    "[720,336,336]",
                    "[720,336,168]",
                    "[720,168,48]",
                    "[720,168,24]",
                ],
            ),
            flag("dataset.timeenc", [1]), # need to use timeenc 1
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]),
            flag("model.n_layers", [1, 2]), # still sweeping both
            flag("model.dropout", [0.25, 0.3]),
            flag("optimizer.lr", [0.01, 0.004]), # adding lr sweep
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

"""
--data ETTh2 
### M
48,48,24
96,96,48
336,336,168
336,168,336
720,336,720

### S
48,48,24
96,96,48
336,336,168
336,168,336
336,336,720
"""

def etth2_sweep_M(): # launched # relaunched

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("+experiment", ["s3-informer-etth"]),
            flag("decoder.mode", ["last"]), # "pool" is terrible
            flag(
                "dataset.size",
                [
                    "[720,336,720]",
                    "[336,168,336]",
                    "[336,336,168]",
                    "[96,96,48]",
                    "[48,48,24]",
                ],
            ),
            flag("dataset.timeenc", [0]), # 1 doesn't help
            flag("dataset.variant", [1]),
            flag("dataset.features", ["M"]),
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.25, 0.3]), # high dropout helps 
            flag("optimizer.lr", [0.01, 0.004]), # adding lr sweep
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

def etth2_sweep_S(): # launched # relaunched

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("+experiment", ["s3-informer-etth"]),
            flag("decoder.mode", ["last"]), # "pool" is terrible
            flag(
                "dataset.size",
                [
                    "[336,336,720]",
                    "[336,168,336]",
                    "[336,336,168]",
                    "[96,96,48]",
                    "[48,48,24]",
                ],
            ),
            flag("dataset.timeenc", [1]), # need to use timeenc 1
            flag("dataset.variant", [1]),
            flag("dataset.features", ["S"]),
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.25, 0.3]), # high dropout helps
            flag("optimizer.lr", [0.01, 0.004]), # adding lr sweep
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

"""
--data ETTm1
### M
672,96,24
96,48,48
384,384,96
672,288,288
672,384,672

### S
96,48,24
96,48,48
384,384,96
384,384,288
384,384,672
"""

def ettm1_sweep_M(): # launched # relaunched

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("+experiment", ["s3-informer-ettm"]),
            flag("decoder.mode", ["last"]), # "pool" is terrible
            flag(
                "dataset.size",
                [
                    "[672,384,672]",
                    "[672,288,288]",
                    "[384,384,96]",
                    "[96,48,48]",
                    "[672,96,24]",
                ],
            ),
            flag("dataset.timeenc", [0]),
            flag("dataset.variant", [0]), # only variant 0 is reported in the Informer paper
            flag("dataset.features", ["M"]), 
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.2, 0.25, 0.3]),
            flag("optimizer.lr", [0.01, 0.004]), # adding lr sweep
            flag("trainer.val_check_interval", [0.1]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

def ettm1_sweep_S(): # launched # relaunched

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("+experiment", ["s3-informer-ettm"]),
            flag("decoder.mode", ["last"]), # "pool" is terrible
            flag(
                "dataset.size",
                [
                    "[384,384,672]",
                    "[384,384,288]",
                    "[384,384,96]",
                    "[96,48,48]",
                    "[96,48,24]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]), # only variant 0 is reported in the Informer paper
            flag("dataset.features", ["S"]),
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.2, 0.25, 0.3]),
            flag("optimizer.lr", [0.01, 0.004]), # adding lr sweep
            flag("trainer.val_check_interval", [0.1]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

"""
--data ECL
Informer repo doesn't report the hyperparameters for this dataset.
- From the paper, we know the pred_lens for the 5 settings.
- The paper sweeps seq_len from 24 upto 720, and we simply use 24 for all settings, since 
    - gives us the biggest disadvantage 
    - allows our evaluation to be over all windows they evaluate on, with potentially extra ones

24,24,48
24,24,168
24,24,336
24,24,720
24,24,960
"""

def ecl_sweep_M(): # launched  # relaunched

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("+experiment", ["s3-informer-ecl"]),
            flag("decoder.mode", ["last"]),
            flag(
                "dataset.size",
                [
                    "[24,24,960]",
                    "[24,24,720]",
                    "[24,24,336]",
                    "[24,24,168]",
                    "[24,24,48]",
                ],
            ),
            flag("dataset.timeenc", [0, 1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["M"]), 
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.2, 0.25, 0.3]),
            flag("optimizer.lr", [0.01, 0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

def ecl_sweep_S(): # launched # relaunched

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("+experiment", ["s3-informer-ecl"]),
            flag("decoder.mode", ["last"]),
            flag(
                "dataset.size",
                [
                    "[24,24,960]",
                    "[24,24,720]",
                    "[24,24,336]",
                    "[24,24,168]",
                    "[24,24,48]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]), 
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.2, 0.25, 0.3]),
            flag("optimizer.lr", [0.01, 0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep


"""
--data WTH
### M
168,168,24
96,96,48
336,168,168
720,168,336
720,336,720

### S
720,168,24
720,168,48
168,168,168
336,336,336
720,336,720
"""

def weather_sweep_M(): # launched  # relaunched

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("+experiment", ["s3-informer-weather"]),
            flag("decoder.mode", ["last"]),
            flag(
                "dataset.size",
                [
                    "[720,336,720]",
                    "[720,168,336]",
                    "[336,168,168]",
                    "[96,96,48]",
                    "[168,168,24]",
                ],
            ),
            flag("dataset.timeenc", [0, 1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["M"]),
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.2, 0.25, 0.3]),
            flag("optimizer.lr", [0.01, 0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

def weather_sweep_S(): # launched

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("+experiment", ["s3-informer-weather"]),
            flag("decoder.mode", ["last"]),
            flag(
                "dataset.size",
                [
                    "[720,336,720]",
                    "[336,336,336]",
                    "[168,168,168]",
                    "[720,168,48]",
                    "[720,168,24]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]),
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.2, 0.25, 0.3]),
            flag("optimizer.lr", [0.01, 0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep


