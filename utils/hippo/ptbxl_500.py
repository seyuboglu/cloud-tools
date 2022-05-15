from ..config import *


def ptbxl500_superdiag_globalsweep():

    sweep = prod(
        [
            flag("train.seed", [0, 1, 2]),
            flag("pipeline", ["ptbxl"]),
            flag("dataset.ctype", ["superdiagnostic"]),
            flag("dataset.sampling_rate", [500]),
            flag("model", ["s4"]),
            flag("optimizer.lr", [0.01, 0.02]),
            flag("+model.layer.lr_dt", ["'${optimizer.lr}'"]),
            flag("model.prenorm", [False]),
            flag("model.dropout", [0.1, 0.2]),
            flag("optimizer.weight_decay", [0.05, 0.1]),
            flag("model.d_model", [256]),
            flag("model.n_layers", [6]),
            flag("loader.batch_size", [32]),
            flag("trainer.max_epochs", [200]),
            flag("model.layer.n_ssm", [256]),
            flag("model.layer.measure", ["legs"]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [True]),
            flag("model.norm", ["layer"]),
            flag("model.prenorm", [False]),
            flag("decoder.mode", ["pool"]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [200]),
            flag("scheduler.warmup_t", [5]),
        ]
    )

    return sweep


def ptbxl500_subdiag_globalsweep():

    sweep = prod(
        [
            flag("train.seed", [0, 1, 2]),
            flag("pipeline", ["ptbxl"]),
            flag("dataset.ctype", ["subdiagnostic"]),
            flag("dataset.sampling_rate", [500]),
            flag("model", ["s4"]),
            flag("optimizer.lr", [0.01, 0.02]),
            flag("+model.layer.lr_dt", ["'${optimizer.lr}'"]),
            flag("model.prenorm", [False]),
            flag("model.dropout", [0.1, 0.2]),
            flag("optimizer.weight_decay", [0.05, 0.1]),
            flag("model.d_model", [256]),
            flag("model.n_layers", [6]),
            flag("loader.batch_size", [32]),
            flag("trainer.max_epochs", [200]),
            flag("model.layer.n_ssm", [256]),
            flag("model.layer.measure", ["legs"]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [True]),
            flag("model.norm", ["layer"]),
            flag("model.prenorm", [False]),
            flag("decoder.mode", ["pool"]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [200]),
            flag("scheduler.warmup_t", [5]),
        ]
    )

    return sweep


def ptbxl500_diag_globalsweep():

    sweep = prod(
        [
            flag("train.seed", [0, 1, 2]),
            flag("pipeline", ["ptbxl"]),
            flag("dataset.ctype", ["diagnostic"]),
            flag("dataset.sampling_rate", [500]),
            flag("model", ["s4"]),
            flag("optimizer.lr", [0.01, 0.02]),
            flag("+model.layer.lr_dt", ["'${optimizer.lr}'"]),
            flag("model.prenorm", [False]),
            flag("model.dropout", [0.1, 0.2]),
            flag("optimizer.weight_decay", [0.05, 0.1]),
            flag("model.d_model", [256]),
            flag("model.n_layers", [6]),
            flag("loader.batch_size", [32]),
            flag("trainer.max_epochs", [200]),
            flag("model.layer.n_ssm", [256]),
            flag("model.layer.measure", ["legs"]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [True]),
            flag("model.norm", ["layer"]),
            flag("model.prenorm", [False]),
            flag("decoder.mode", ["pool"]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [200]),
            flag("scheduler.warmup_t", [5]),
        ]
    )

    return sweep


def ptbxl500_form_globalsweep():

    sweep = prod(
        [
            flag("train.seed", [0, 1, 2]),
            flag("pipeline", ["ptbxl"]),
            flag("dataset.ctype", ["form"]),
            flag("dataset.sampling_rate", [500]),
            flag("model", ["s4"]),
            flag("optimizer.lr", [0.01, 0.02]),
            flag("+model.layer.lr_dt", ["'${optimizer.lr}'"]),
            flag("model.prenorm", [False]),
            flag("model.dropout", [0.1, 0.2]),
            flag("optimizer.weight_decay", [0.05, 0.1]),
            flag("model.d_model", [256]),
            flag("model.n_layers", [6]),
            flag("loader.batch_size", [32]),
            flag("trainer.max_epochs", [200]),
            flag("model.layer.n_ssm", [256]),
            flag("model.layer.measure", ["legs"]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [True]),
            flag("model.norm", ["layer"]),
            flag("model.prenorm", [False]),
            flag("decoder.mode", ["pool"]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [200]),
            flag("scheduler.warmup_t", [5]),
        ]
    )

    return sweep


def ptbxl500_rhythm_globalsweep():

    sweep = prod(
        [
            flag("train.seed", [0, 1, 2]),
            flag("pipeline", ["ptbxl"]),
            flag("dataset.ctype", ["rhythm"]),
            flag("dataset.sampling_rate", [500]),
            flag("model", ["s4"]),
            flag("optimizer.lr", [0.01, 0.02]),
            flag("+model.layer.lr_dt", ["'${optimizer.lr}'"]),
            flag("model.prenorm", [False]),
            flag("model.dropout", [0.1, 0.2]),
            flag("optimizer.weight_decay", [0.05, 0.1]),
            flag("model.d_model", [256]),
            flag("model.n_layers", [6]),
            flag("loader.batch_size", [32]),
            flag("trainer.max_epochs", [200]),
            flag("model.layer.n_ssm", [256]),
            flag("model.layer.measure", ["legs"]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [True]),
            flag("model.norm", ["layer"]),
            flag("model.prenorm", [False]),
            flag("decoder.mode", ["pool"]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [200]),
            flag("scheduler.warmup_t", [5]),
        ]
    )

    return sweep


def ptbxl500_all_globalsweep():

    sweep = prod(
        [
            flag("train.seed", [0, 1, 2]),
            flag("pipeline", ["ptbxl"]),
            flag("dataset.ctype", ["all"]),
            flag("dataset.sampling_rate", [500]),
            flag("model", ["s4"]),
            flag("optimizer.lr", [0.01, 0.02]),
            flag("+model.layer.lr_dt", ["'${optimizer.lr}'"]),
            flag("model.prenorm", [False]),
            flag("model.dropout", [0.1, 0.2]),
            flag("optimizer.weight_decay", [0.05, 0.1]),
            flag("model.d_model", [256]),
            flag("model.n_layers", [6]),
            flag("loader.batch_size", [32]),
            flag("trainer.max_epochs", [200]),
            flag("model.layer.n_ssm", [256]),
            flag("model.layer.measure", ["legs"]),
            flag("model.layer.postact", ["glu"]),
            flag("model.layer.bidirectional", [True]),
            flag("model.norm", ["layer"]),
            flag("model.prenorm", [False]),
            flag("decoder.mode", ["pool"]),
            flag("scheduler", ["timm_cosine"]),
            flag("scheduler.t_initial", [200]),
            flag("scheduler.warmup_t", [5]),
        ]
    )

    return sweep


# lzip(
#     [
#         flag(
#             "model.layer.measure",
#             [
#                 "legs",
#                 "fourier",
#                 "fourier_diag",
#                 "hippo",
#                 "all",
#                 "random",
#             ],
#         ),
#         flag("+model.layer.rank_weight", [1, 1, 1, 1, 1, 0]),
#     ]
# ),
