from utils.config import flag, chain, prod, lzip

def samplernn_repro():
    sweep = prod([
        flag("experiment", ['samplernn-qautomusic']),
        prod([
            flag("dataset.quantization", ['linear', 'mu-law']),
        ]),
    ])

    return sweep


def samplernn_general_repro():
    # This failed to be a good reproduction of the original model!
    sweep = prod([
        flag("experiment", ['samplernn-qautomusic']),
        flag("model.reproduce", [False]),
        prod([
            flag("dataset.quantization", ['linear', 'mu-law']),
        ]),
    ])

    return sweep

def samplernn_repro():
    sweep = prod([
        flag("experiment", ['samplernn-qautomusic']),
        prod([
            flag("dataset.quantization", ['linear', 'mu-law']),
        ]),
    ])

    return sweep

# SampleRNN with S4
# python -m train experiment=samplernn-qautomusic model.layer=s4  wandb=null loader.batch_size=1 dataset.sample_len=131135 train.state.chunk_len=131072


# CUDA_VISIBLE_DEVICES=3 python -m train experiment=samplernn-qautomusic wandb.group=samplernn-repro-11-29 dataset.quantization=linear dataset.path=/scr-ssd/mercury/hazy/hippo/data/music_data &
# CUDA_VISIBLE_DEVICES=3 python -m train experiment=samplernn-qautomusic wandb.group=samplernn-repro-11-29 dataset.quantization=mu-law dataset.path=/scr-ssd/mercury/hazy/hippo/data/music_data &