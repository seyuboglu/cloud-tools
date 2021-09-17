import argparse
import subprocess
import itertools
import functools
import importlib
import datetime
import numpy as np
import inspect

chain = lambda l: list(itertools.chain(*l))

def _escape(k, v):
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list):
        return f"'{k}={v}'"
    else:
        return f"{k}={v}"

def flag(k, vs):
    """
    flag('seed', [0,1,2]) returns [['seed=0'], ['seed=1'], ['seed=2']]
    """
    return [[_escape(k, v)] for v in vs]

def pref(prefix, L):
    pref_fn = lambda s: prefix+'.'+s
    # return map(functools.partial(map, pref_fn), prod(L))
    return [[pref_fn(s) for s in l] for l in prod(L)]

def prod(L):
    p = itertools.product(*L)
    return list(map(chain, p))

def lzip(L):
    if len(L) == 0:
        return []
    assert np.all(np.array(list(map(len, L))) == len(L[0])), "zip: unequal list lengths"

    out = map(chain, zip(*L))
    return list(out)