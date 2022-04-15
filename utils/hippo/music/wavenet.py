from utils.config import flag, chain, prod, lzip

def youtubemix_1():
    sweep = prod([
        flag("experiment", ['wavenet-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("dataset.pad_len", [4093]),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/']),
        flag("loader.batch_size", [1]),
        flag("trainer.max_epochs", [500]),
        flag("scheduler.patience", [5]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        flag("optimizer.lr", [0.001]),
        flag("model.skip_channels", [512, 1024]),
    ])

    return sweep

def sc09_1():
    sweep = prod([
        flag("experiment", ['wavenet-qautomusic']),
        flag("dataset", ['sc09']),
        flag("dataset.quantization", ['mu-law']),
        flag("dataset.pad_len", [4093]),
        flag("loader.batch_size", [8]),
        flag("trainer.max_epochs", [500]),
        flag("scheduler.patience", [5]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        flag("optimizer.lr", [0.001]),
        flag("model.skip_channels", [512, 1024]),
        flag("decoder.mode", ['ragged']),
    ])

    return sweep

def beethoven_1():
    sweep = prod([
        flag("experiment", ['wavenet-qautomusic']),
        flag("dataset.path", ['/home/workspace/hippo/data/beethoven/']),
        flag("dataset.sample_len", [128000]),
        flag("dataset.quantization", ['linear']),
        flag("dataset.pad_len", [4093]),
        flag("loader.batch_size", [1]),
        flag("trainer.max_epochs", [500]),
        flag("scheduler.patience", [5]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        flag("optimizer.lr", [0.001]),
        flag("model.skip_channels", [512, 1024]),
        flag("decoder.mode", ['ragged']),
    ])

    return sweep

def youtubemix_2():
    sweep = prod([
        flag("experiment", ['wavenet-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("dataset.pad_len", [4093]),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/']),
        flag("loader.batch_size", [1]),
        flag("trainer.max_epochs", [500]),
        flag("scheduler.patience", [5]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        flag("optimizer.lr", [0.001]),
        flag("model.skip_channels", [512]),
        flag("model.residual_channels", [128, 256]),
    ])

    return sweep