from utils.config import *

###############################################################################
#                                                                             #
#                              Informer Datasets                              #
#                                                                             #
###############################################################################

def ett1h_longest_common_flags():

    sweep = prod(
        [
            flag("experiment", ["s4-informer-etth"]),
            flag(
                "dataset.size",
                [
                    "[720,336,720]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]),
            flag("task.metrics", ["[mse,mae]"]),
            flag("dataset.target", ["OT"]),
            flag("trainer.max_epochs", [5]),
            flag("loader.batch_size", [50]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [5]),
            flag("scheduler.warmup_t", [0]),
        ]
    )
    return sweep

def ett2h_longest_common_flags():

    sweep = prod(
        [
            flag("experiment", ["s4-informer-etth"]),
            flag(
                "dataset.size",
                [
                    "[336,336,720]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [1]),
            flag("dataset.features", ["S"]),
            flag("task.metrics", ["[mse,mae]"]),
            flag("dataset.target", ["OT"]),
            flag("trainer.max_epochs", [10]),
            flag("loader.batch_size", [50]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [10]),
            flag("scheduler.warmup_t", [0]),
        ]
    )
    return sweep

def ettm_longest_common_flags():
    
    sweep = prod(
        [
            flag("experiment", ["s4-informer-ettm"]),
            flag(
                "dataset.size",
                [
                    "[672,672,672]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]),
            flag("task.metrics", ["[mse,mae]"]),
            flag("dataset.target", ["OT"]),
            flag("trainer.max_epochs", [5]),
            flag("loader.batch_size", [50]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [5]),
            flag("scheduler.warmup_t", [0]),
        ]
    )
    return sweep


def ett1h_shorter_common_flags():

    sweep = prod(
        [
            flag("experiment", ["s4-informer-etth"]),
            flag(
                "dataset.size",
                [
                    "[720,336,336]",
                    "[720,336,168]",
                    "[720,168,48]",
                    "[720,168,24]",
                ],
            ),
            flag("dataset.timeenc", [1]),
            flag("dataset.variant", [0]),
            flag("dataset.features", ["S"]),
            flag("task.metrics", ["[mse,mae]"]),
            flag("dataset.target", ["OT"]),
            flag("trainer.max_epochs", [5]),
            flag("loader.batch_size", [50]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [5]),
            flag("scheduler.warmup_t", [0]),
        ]
    )
    return sweep


def etth1_global_1():
    # bidirectional sweep
    # bidirectional vs. unidirectional is very close
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("model.dropout", [0.30]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [True, False]),
            flag("model.layer.n_ssm", [128]),
            flag("optimizer.weight_decay", [0.0]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])

def etth1_global_2():
    # lr_dt sweep
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("model.dropout", [0.30]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.001, 0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("optimizer.weight_decay", [0.0]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])

def etth1_global_3():
    # Wd + dropout sweep
    # need dropout >= 0.2 (lower dropout generalizes worse)
    # Best combinations (trends on both measures are very similar):
    # dropout 0.2, wd 0.1, 0.2
    # dropout 0.3, wd 0.05
    # best val: dropout 0.2, wd 0.2, fourier_diag
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.1, 0.2, 0.3]),
            flag("optimizer.weight_decay", [0.05, 0.10, 0.20]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])


def etth1_global_4():
    # d_state sweep
    # d_state 64 still works the best, although 16 is not bad (256 seems worse)
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.layer.d_state", [16, 64, 256]),
            flag("model.dropout", [0.2]),
            flag("optimizer.weight_decay", [0.20]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])

def etth1_global_5():
    # sweep normalization, prenorm
    # prenorm False works better
    # batch norm is really bad, layer norm is better
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.norm", ['batch', 'layer']),
            flag("model.prenorm", [True, False]),
            flag("model.dropout", [0.2]),
            flag("optimizer.weight_decay", [0.20]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])


def etth1_global_6():
    # sweep measures and best (dropout, wd) settings
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            lzip([
                flag("model.dropout", [0.2, 0.2, 0.3]),
                flag("optimizer.weight_decay", [0.20, 0.10, 0.05]),
            ]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "legsd",
                            "hippo",
                            "fourier",
                            "fourier_diag",
                            "fourier_decay",
                            "fourier_old",
                            "random",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1, 1, 1, 1, 1, 1, 0]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])

def etth1_global_7():
    # sweep measures and a few other high dropout settings
    # dropout 0.4 seems good on test, but val is higher
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            lzip([
                flag("model.dropout", [0.3, 0.4]),
                flag("optimizer.weight_decay", [0.0, 0.0]),
            ]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "legsd",
                            "hippo",
                            "fourier",
                            "fourier_diag",
                            "fourier_decay",
                            "fourier_old",
                            "random",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1, 1, 1, 1, 1, 1, 0]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_longest_common_flags()])


def etth1_shorter_sweep():
    # sweep measures and a few other high dropout settings
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            lzip([
                flag("model.dropout", [0.2, 0.2, 0.3]),
                flag("optimizer.weight_decay", [0.20, 0.10, 0.05]),
            ]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "legsd",
                            "hippo",
                            "fourier",
                            "fourier_diag",
                            "fourier_decay",
                            "fourier_old",
                            "random",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1, 1, 1, 1, 1, 1, 0]),
                ]
            ),
        ]
    )
    return prod([sweep, ett1h_shorter_common_flags()])

def etth2_global_1():
    # sweep measures and a few other high dropout settings
    # 0.3 dropout / 0.2 wd seems best
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.1, 0.2, 0.3]),
            flag("optimizer.weight_decay", [0.05, 0.10, 0.20]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett2h_longest_common_flags()])


def etth2_global_2():
    sweep = prod(
        [
            flag("train.seed", [5, 6, 7, 8, 9]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.3]),
            flag("optimizer.weight_decay", [0.50, 1.0]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([sweep, ett2h_longest_common_flags()])

def etth2_global_3():
    # hippo does best
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.3]),
            flag("optimizer.weight_decay", [0.2]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "legsd",
                            "hippo",
                            "fourier",
                            "fourier_diag",
                            "fourier_decay",
                            "fourier_old",
                            "random",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1, 1, 1, 1, 1, 1, 0]),
                ]
            ),
        ]
    )
    return prod([sweep, ett2h_longest_common_flags()])


def ettm_global_1():
    sweep = prod(
        [
            flag("train.seed", [0, 1, 2, 3, 4]),
            flag("model.n_layers", [4]),
            flag("optimizer.lr", [0.01]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [False]),
            flag("+model.layer.lr_dt", [0.01]),
            flag("model.layer.n_ssm", [128]),
            flag("model.dropout", [0.1, 0.2, 0.3]),
            flag("optimizer.weight_decay", [0.05, 0.10, 0.20]),
            lzip(
                [
                    flag(
                        "model.layer.measure",
                        [
                            "legs",
                            "fourier_diag",
                        ],
                    ),
                    flag("+model.layer.rank_weight", [1, 1]),
                ]
            ),
        ]
    )
    return prod([ettm_longest_common_flags(), sweep])
