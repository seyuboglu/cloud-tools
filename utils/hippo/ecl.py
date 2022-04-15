from utils.config import *

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


def ecl_sweep_S():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[24,24,960]",
                    "[48,48,960]",
                    "[96,96,960]",
                    "[336,336,960]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]), 
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.2, 0.25, 0.3]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            flag("model.layer.measure", ['legt', 'legs', 'fourier']),

            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep


def ecl_sweep_S_2():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[24,24,960]",
                    "[336,336,960]",
                    "[960,960,960]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]), 
            flag("model.n_layers", [1, 2, 4]),
            flag("model.dropout", [0.25, 0.3, 0.35]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            lzip([
                flag("model.layer.measure", ['legt', 'legs', 'fourier']),
                flag("model.layer.rank", [2, 1, 1]),
            ]),
            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep


def ecl_sweep_S_3():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-informer-ecl"]),
            flag("model", ['unet']),
            flag(
                "dataset.size",
                [
                    "[24,24,960]",
                    "[336,336,960]",
                    "[960,960,960]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]), 

            flag("model.expand", [2]),
            flag("model.ff", [2]),
            flag("model.pool", [[4], [4, 4], [2], [2, 2]]),

            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.25, 0.3, 0.35]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            lzip([
                flag("model.layer.measure", ['legt', 'legs', 'fourier']),
                flag("model.layer.rank", [2, 1, 1]),
            ]),
            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep



def ecl_sweep_S_168():

    sweep = prod(
        [
            flag("train.seed", [0, 1, 2]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[24,24,168]",
                    "[336,336,168]",
                    "[960,960,168]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]), 
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.25, 0.3, 0.35]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            lzip([
                flag("model.layer.measure", ['fourier']),
                flag("model.layer.rank", [1]),
            ]),
            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep


def ecl_sweep_S_168_2():

    sweep = prod(
        [
            flag("train.seed", [0, 1, 2]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[24,24,168]",
                    "[336,336,168]",
                    "[960,960,168]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]), 
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.25, 0.3, 0.35]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            lzip([
                flag("model.layer.measure", ['legs']),
                flag("model.layer.rank", [1]),
            ]),
            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep


def ecl_sweep_S_168_3():

    sweep = prod(
        [
            flag("train.seed", [0, 1, 2]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[96,96,168]",
                    "[168,168,168]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]), 
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.25, 0.3, 0.35]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            lzip([
                flag("model.layer.measure", ['fourier']),
                flag("model.layer.rank", [1]),
            ]),
            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

def ecl_sweep_S_168_4():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[168,168,168]",
                    "[336,336,168]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]), 
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.25, 0.3, 0.35]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True, False]),
            flag("model.layer.postact", ['glu', 'null']),
            lzip([
                flag("model.layer.measure", ['fourier']),
                flag("model.layer.rank", [1]),
            ]),
            flag("optimizer.lr", [0.004]),
            flag("scheduler", ['step']),
            flag("train.interval", ['epoch']),
            flag("scheduler.gamma", [0.25, 0.50, 0.75]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep


def ecl_sweep_M():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[24,24,960]",
                ],
            ),
            flag("dataset.timeenc", [0, 1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["M"]), 
            
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.2, 0.25, 0.3]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            flag("model.layer.measure", ['fourier']),

            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep


def ecl_sweep_M_2():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[336,336,960]",
                    "[960,960,960]",
                ],
            ),
            flag("dataset.timeenc", [0, 1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["M"]), 
            
            flag("model.n_layers", [1, 2]),
            flag("model.dropout", [0.2, 0.25, 0.3]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            flag("model.layer.measure", ['fourier']),

            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

def ecl_sweep_M_3():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[336,336,960]",
                    "[960,960,960]",
                ],
            ),
            flag("dataset.timeenc", [0, 1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["M"]), 
            
            flag("model.n_layers", [4]),
            flag("model.dropout", [0.2, 0.25, 0.3]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            flag("model.layer.measure", ['fourier']),

            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep


def ecl_sweep_M_4():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[336,336,960]",
                    "[960,960,960]",
                ],
            ),
            flag("dataset.timeenc", [0, 1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["M"]), 
            
            flag("model.n_layers", [1, 2, 4, 6]),
            flag("model.dropout", [0.1, 0.2, 0.25, 0.3]),
            flag("model.prenorm", [True]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            flag("model.layer.measure", ['fourier']),

            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep

def ecl_sweep_M_5():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-informer-ecl"]),
            flag(
                "dataset.size",
                [
                    "[336,336,960]",
                    "[960,960,960]",
                ],
            ),
            flag("dataset.timeenc", [0, 1]),
            flag("dataset.variant", [0]),
            lzip([
                flag("dataset.eval_mask", [True, False, True]),
                flag("dataset.eval_stamp", [True, True, False]),
            ]),
            
            flag("dataset.features", ["M"]), 
            
            flag("model.n_layers", [1, 2, 4, 6]),
            flag("model.dropout", [0.1, 0.2, 0.25, 0.3]),
            flag("model.prenorm", [True]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            flag("model.layer.measure", ['fourier']),

            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep



def ecl_sweep_M_6():

    sweep = prod(
        [
            flag("train.seed", [0]),
            flag("experiment", ["s4-informer-ecl"]),
            flag("model", ['unet', 'sashimi']),
            flag(
                "dataset.size",
                [
                    "[336,336,960]",
                    "[960,960,960]",
                ],
            ),
            flag("dataset.timeenc", [0]),
            flag("dataset.variant", [0]),
            flag("model.expand", [2]),
            flag("model.ff", [2]),
            flag("model.pool", [[4], [4, 4], [2], [2, 2]]),
            
            flag("dataset.features", ["M"]), 
            
            flag("model.n_layers", [1, 2, 4]),
            flag("model.dropout", [0.1, 0.3]),
            flag("model.prenorm", [True]),
            
            flag("model.layer.hurwitz", [True]),
            flag("model.layer.trainable.A", [True]),
            flag("model.layer.trainable.B", [True]),
            flag("model.layer.trainable.P", [True]),
            flag("model.layer.trainable.dt", [True]),
            flag("model.layer.tie_state", [True]),
            flag("model.layer.postact", ['glu']),
            flag("model.layer.measure", ['fourier']),

            flag("optimizer.lr", [0.004]),
            flag("task.metrics", ["[mse,mae]"]),
        ]
    )

    return sweep