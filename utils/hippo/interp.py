from ..config import *

def cifar_sweep_1():

    sweep = prod([
        flag('experiment', [ 's4-cifar-diagnostic', ]),
        
        flag('model.layer.measure', ['legs', 'legsd', 'random']),
        flag('+model.layer.quadrature', ['trapezoid', 'simpson', None]),        
    ])

    return sweep

def cifar_sweep_1_longer():

    sweep = prod([
        flag('experiment', [ 's4-cifar-diagnostic', ]),
        
        flag('model.layer.measure', ['legs', 'legsd', 'random']),
        flag('+model.layer.quadrature', ['trapezoid', 'simpson', None]),
        flag("scheduler.num_training_steps", [200000]),
        flag("trainer.max_epochs", [200]),
    ])

    return sweep

def cifar_sweep_foh():

    sweep = prod([
        flag('experiment', [ 's4-cifar-diagnostic']),
        flag('model.layer.measure', ['legsd']),
        flag('model.layer.mode', ['diag']),
        flag('+model.layer.disc', ['zoh', 'foh']),
    ])

    return sweep

def cifar_sweep_cheby():

    sweep = prod([
        flag('experiment', [ 's4-cifar-diagnostic']),
        flag('model.layer.measure', ['legsd']),
        flag('model.layer.mode', ['diag']),
        flag('+model.layer.disc', ['zoh', 'chebyshev', 'chebyshev-approx']),
        flag('model.layer.bidirectional', [False]),
        flag('loader.batch_size', [16]),
    ])

    return sweep

def cifar_sweep_2():

    prod([
        flag('model.layer.mode', ['diag']),
        flag('+model.layer.disc', ['zoh', 'bilinear']),
        flag('+model.layer.measure_args.scaling', ['linear', 'inverse']),
        flag('+model.layer.rank_weight', [0.0]),
        flag('+model.layer.measure_args.normalize', [True]),
        flag('+model.layer.quadrature', ['trapezoid', 'simpson', None]),
    ]),

    # sweep = prod([
    #     flag('experiment', [ 's4-cifar-diagnostic', ]),
    #     chain([
    #         flag('model.layer.measure', ['legs', 'legsd']),
    #         prod([
    #             flag('model.layer.mode', ['diag']),
    #             flag('+model.layer.disc', ['zoh', 'bilinear', 'dss']),
    #             flag('model.layer.measure', ['random']),
    #             flag('+model.layer.measure_args.scaling', ['linear', 'inverse']),
    #             flag('+model.layer.rank_weight', [0.0]),
    #             flag('+model.layer.measure_args.normalize', [True]),
    #         ]),
    #     ]),
    # ])
