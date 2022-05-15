from ..config import *


def arima_s4_sweep():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [0, 0, 0, 0, 0, 1, 2, 3, 5, 10, 20]),
                flag("dataset.d", [0, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0]),
                flag("dataset.q", [0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0]),
                flag("dataset.lag", [1, 1, 1, 2, 2, 1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.horizon", [10]),
        ]
    )

    return sweep

def arima_s4_sweep_01k():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            flag("dataset.d", [1]),
            lzip([
                flag("dataset.q", [0, 1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [0, 1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.horizon", [10]),
        ]
    )

    return sweep

def arima_s4_sweep_k00():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.d", [0]),
            flag("dataset.q", [0]),
            lzip([
                flag("dataset.p", [1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.horizon", [10]),
        ]
    )

    return sweep

def single_arima_s4_sweep_k00():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.d", [0]),
            flag("dataset.q", [0]),
            lzip([
                flag("dataset.p", [1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [10]),
        ]
    )

    return sweep

def single_arima_s4_sweep_0dq():
    # Can we learn differencing?
    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            flag("dataset.d", [1, 2]),
            flag("model.n_layers", [1, 2, 4]),
            lzip([
                flag("dataset.q", [0, 1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [0, 1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [10]),
        ]
    )

    return sweep

def single_arima_s4_sweep_k00_20():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.d", [0]),
            flag("dataset.q", [0]),
            flag("dataset.p", [20]),
            lzip([
                flag("dataset.lag", [20]),
            ]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]), # set horizon to 1
        ]
    )

    return sweep

def single_arima_s4_sweep_k00_20_big():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("model.n_layers", [1, 2, 4]), # may need a bigger model to learn for longer horizon
            flag("dataset.d", [0]),
            flag("dataset.q", [0]),
            flag("dataset.p", [20]),
            lzip([
                flag("dataset.lag", [20]),
            ]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [10]),
        ]
    )
    return sweep

def single_arima_s4_sweep_0dq_020():
    # Can we learn differencing?
    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            flag("dataset.d", [2]),
            flag("dataset.q", [0]),
            flag("dataset.lag", [1, 2, 4, 8, 16, 32]),
            flag("model.n_layers", [1, 2, 4]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
        ]
    )

    return sweep

def single_arima_s4_sweep_0dq_020_measures():
    # Can we learn differencing?
    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            flag("dataset.d", [2]),
            flag("dataset.q", [0]),
            flag("dataset.lag", [1, 2, 4]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.measure", ['random_lin', 'random_inv', 'fourier', 'legs']),
        ]
    )

    return sweep

# Layer norm and activation messes up perofrmance

def single_arima_s4_sweep_0dq_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            flag("dataset.d", [1, 2, 3]),
            flag("dataset.q", [0]),
            flag("dataset.lag", [1, 2, 3]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep

def single_arima_s4_sweep_0dq_sweep_2():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            flag("dataset.p", [0]),
            lzip([
                flag("dataset.d", [1, 2, 3]),
                flag("dataset.lag", [1, 2, 3]),
            ]),
            flag("dataset.q", [0]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null']),
            flag("model.norm", ['null']),
            flag("model.layer.measure", ['random-linear', 'fourier', 'random-inv', 'legs', 'legt']),
        ]
    )

    return sweep

def single_arima_s4_sweep_p00_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [0, 1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [0, 1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.d", [0]),
            flag("dataset.q", [0]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep

def single_arima_s4_sweep_00q_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.q", [0, 1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.d", [0]),
            flag("dataset.p", [0]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep


def single_arima_s4_sweep_p0q_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [1, 2, 3, 5, 10, 20]),
                flag("dataset.q", [1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.d", [0]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep


def single_arima_s4_sweep_p1q_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [1, 2, 3, 5, 10, 20]),
                flag("dataset.q", [1, 2, 3, 5, 10, 20]),
                flag("dataset.lag", [1, 2, 3, 5, 10, 20]),
            ]),
            flag("dataset.d", [1]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep

def single_arima_s4_sweep_arima_ets_sweep_1():

    sweep = prod(
        [
            flag("experiment", ["s4-synthetic-arima"]),
            lzip([
                flag("dataset.p", [0, 0, 1]),
                flag("dataset.q", [1, 2, 1]),
                flag("dataset.lag", [1, 2, 2]),
            ]),
            flag("dataset.d", [1]),
            flag("model.n_layers", [1]),
            flag("dataset.n_ts", [1]),
            flag("dataset.nobs_per_ts", [1000]),
            flag("dataset.horizon", [1]),
            flag("model.layer.activation", ['null', 'gelu']),
            flag("model.norm", ['null', 'layer']),
            flag("model.layer.measure", ['random-linear']),
        ]
    )

    return sweep


# Sweep ideas
# Effect of d_state
# Sweep lag vs. order of ARIMA
# Effect of differencing
# Effect of constant


# ARIMA(0, 0, 0) (white noise) and ARIMA(1, 0, 0) are fit perfectly (error equals the std of the noise)
# The fit for ARIMA(0, 1, k) models improves from 0 -> 1 -> 2, and seems rather high for ARIMA(0, 1, 0)
    # TODO: try differencing 
# The fit for ARIMA(0, 2, 2) seems pretty bad, gap is increasing from train -> val -> test
    # TODO: compare to best fit ARIMA(0, 2, 2) model
# The fit for ARIMA(2, 0, 0) has error around 2x the std of the noise
    # TODO: check that this is the expected error -- yes

