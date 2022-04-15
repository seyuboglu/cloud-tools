from utils.config import flag, chain, prod, lzip

def youtubemix():
    sweep = prod([
        flag("experiment", ['samplernn-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['linear', 'mu-law']),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [-1]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
        flag("loader.batch_size", [32]),
    ])

    return sweep

def beethoven():
    sweep = prod([
        flag("experiment", ['samplernn-qautomusic']),
        flag("dataset.path", ['/home/workspace/hippo/data/beethoven/']),
        flag("dataset.sample_len", [128000]),
        flag("dataset.quantization", ['linear']),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        lzip([
            flag("model.n_rnn", [1, 2]),
            flag("model.frame_sizes", [[8, 2, 2], [16, 4]]),
            flag("train.state.overlap_len", [32, 64]),
        ]),
        
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep

def sc09():
    sweep = prod([
        flag("experiment", ['samplernn-qautomusic']),
        flag("dataset", ['sc09']),
        flag("dataset.quantization", ['mu-law']),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        lzip([
            flag("model.n_rnn", [1, 2]),
            flag("model.frame_sizes", [[8, 2, 2], [16, 4]]),
            flag("train.state.overlap_len", [32, 64]),
        ]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep

def youtubemix_2():
    sweep = prod([
        flag("experiment", ['samplernn-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/']),
        flag("dataset.quantization", ['mu-law']),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        lzip([
            flag("model.n_rnn", [1, 2]),
            flag("model.frame_sizes", [[8, 2, 2], [16, 4]]),
            flag("train.state.overlap_len", [32, 64]),
        ]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
        flag("loader.batch_size", [32]),
    ])

    return sweep


def beethoven2():
    sweep = prod([
        flag("experiment", ['samplernn-qautomusic']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/beethoven/']),
        flag("dataset.sample_len", [128000]),
        flag("dataset.quantization", ['linear']),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        lzip([
            flag("model.n_rnn", [3]),
            flag("model.frame_sizes", [[2, 2]]),
            flag("train.state.overlap_len", [4]),
        ]),
        
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep

def sc09_2():
    sweep = prod([
        flag("experiment", ['samplernn-qautomusic']),
        flag("dataset", ['sc09']),
        flag("dataset.quantization", ['mu-law']),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        lzip([
            flag("model.n_rnn", [3]),
            flag("model.frame_sizes", [[2, 2]]),
            flag("train.state.overlap_len", [4]),
        ]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep

def youtubemix_3():
    sweep = prod([
        flag("experiment", ['samplernn-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/']),
        flag("dataset.quantization", ['mu-law']),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        lzip([
            flag("model.n_rnn", [3]),
            flag("model.frame_sizes", [[2, 2]]),
            flag("train.state.overlap_len", [4]),
        ]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
        flag("loader.batch_size", [32]),
    ])

    return sweep

