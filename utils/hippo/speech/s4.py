from utils.config import flag, chain, prod, lzip

def s4_embedding_repro():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        prod([
            flag("dataset.quantization", ['linear', 'mu-law']),
        ]),
    ])

    return sweep
