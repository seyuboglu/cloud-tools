from utils.config import flag, chain, prod, lzip

def s4_youtube_small():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['linear', 'mu-law']),
        flag("model.expand", [1, 2]),
        flag("model.n_layers", [2]),  
        flag("trainer.max_epochs", [500]),
    ])

    return sweep

def s4_youtube_smalltwo():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("model.expand", [2]),
        flag("model.n_layers", [2]),  
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [-1]),
        flag("task.metrics", [['bpb', 'accuracy']]),
        # flag("task.torchmetrics", [['Accuracy@1', 'Accuracy@5', 'Accuracy@10']]), # slows down training
    ])

    return sweep

def s4_youtube_smalltwo_16s():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("model.expand", [2]),
        flag("model.n_layers", [2]),  
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [-1]),
        flag("dataset.sample_len", [262144]),
        flag("loader.batch_size", [1]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
        # flag("task.torchmetrics", [['Accuracy@1', 'Accuracy@5', 'Accuracy@10']]), # slows down training
    ])

    return sweep

def s4_ljspeech_smalltwo():
    sweep = prod([
        flag("experiment", ['s4-ljspeech']),
        flag("dataset.quantization", ['mu-law']),
        flag("model.expand", [2]),
        flag("model.n_layers", [2]),
        flag("model.pool", [[4, 4]]),
        flag("callbacks.model_checkpoint.save_top_k", [-1]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
        flag("loader.batch_size", [1]),
    ])

    return sweep

def s4_youtube_snet_smalltwo():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.n_layers", [2]),  
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [-1]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
        # flag("task.torchmetrics", [['Accuracy@1', 'Accuracy@5', 'Accuracy@10']]), # slows down training
    ])

    return sweep


def s4_youtube_snet_smalltwo_longseq():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.n_layers", [2]),  
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [-1]),
        flag("loader.batch_size", [1]),
        flag("model.dropout", [0.1]),
        flag("dataset.sample_len", [471040]), # 29.5s
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep

def s4_youtube_smalltwo_datavariations():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        lzip([
            flag("dataset", ['youtubemix', 'youtubemix', 'youtubemix-hires']),
            flag("dataset.bits", [10, 12, 8]),
            flag("loader.batch_size", [1, 1, 1]),
            flag("model", ['unet', 'unet', 'snet']),
            flag("model.d_model", [64, 32, 64]),
        ]),
        flag("dataset.quantization", ['mu-law']),
        flag("model.expand", [2]),
        flag("model.n_layers", [2]),  
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [-1]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep


def s4_youtube_snetbigsweep():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        lzip([
            flag("model.expand", [
                2, 2, 
                2, 2, 2, 
                1, 1, 1, 1,
                1,
            ]),
            flag("model.n_layers", [
                2, 4, 
                4, 6, 8, 
                2, 4, 6, 8,
                2,
            ]),
            flag("model.pool", [
                [4, 4, 4], [4, 4, 4], 
                [4, 4], [4, 4], [4, 4], 
                [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4], 
            ]),
            flag("model.ff", [
                4, 4, 
                2, 2, 2, 
                2, 2, 2, 2,
                2,
            ]),
            flag("loader.batch_size", [
                2, 1, 
                2, 1, 1,
                4, 2, 2, 1,
                4,
            ]),
        ]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep



def youtube_snet_paramsweep():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        lzip([
            flag("model.layer.trainable.A", [1, 2, 2, 1, 1, 1]),
            flag("model.layer.trainable.C", [2, 1, 2, 1, 1, 1]),
            flag("model.layer.trainable.dt", [0, 0, 0, 1, 1, 0]),
            flag("model.layer.dt_min", [0.001, 0.001, 0.001, 0.001, 0.0001, 0.0001]),
        ]),
        flag("model.expand", [2]),
        flag("model.n_layers", [2]),
        flag("loader.batch_size", [4]),
        flag("model.ff", [2]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [20]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep



def s4_youtube_snet8layer_resume():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.n_layers", [8]),
        flag("model.pool", [[4, 4]]),
        flag("model.ff", [2]),
        flag("loader.batch_size", [1]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
        flag("trainer.resume_from_checkpoint", ['/home/workspace/hippo/outputs/2022-01-13/08-23-55/checkpoints/val/loss-v8.ckpt'])
    ])

    return sweep

def snet_sc09():
    sweep = prod([
        flag("experiment", ['s4-sc09']),
        lzip([
            flag("model.n_layers", [2, 4, 8]),
            flag("loader.batch_size", [32, 16, 8]),
        ]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def youtube_snet_morelayers():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("dataset.sample_len", [65536]),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.n_layers", [12, 16]),
        flag("model.pool", [[4, 4]]),
        flag("model.ff", [2]),
        flag("loader.batch_size", [1]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def youtube_snet_interp():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.n_layers", [8]),
        flag("model.interp", [2, 4]),
        flag("model.pool", [[4, 4]]),
        flag("model.ff", [2]),
        flag("loader.batch_size", [2]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def youtube_snet_sweep_2():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.n_layers", [2]),
        lzip([
            flag("model.expand", [4, 4, 2]),
            flag("model.ff", [4, 4, 2]),
            flag("model.pool", [[4, 4], [8, 8, 2], [4, 4]]),
            flag("loader.batch_size", [2, 1, 2]),
        ]),
        flag("model.act_pool", ['glu']),
        flag("model.layer.trainable.A", [2]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/'])
    ])

    return sweep

def snet_sc09_bigger():
    sweep = prod([
        flag("experiment", ['s4-sc09']),
        flag("model.n_layers", [8]),
        flag("loader.batch_size", [4]),
        flag("model.expand", [3]),
        flag("model.act_pool", ['glu']),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def snet_sc09_bigger2():
    sweep = prod([
        flag("experiment", ['s4-sc09']),
        flag("model.n_layers", [8]),
        flag("loader.batch_size", [4]),
        flag("model.expand", [3]),
        # flag("model.act_pool", ['glu']), # does this cause nans?
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def snet_sc09_bigger3():
    sweep = prod([
        flag("experiment", ['s4-sc09']),
        flag("model.n_layers", [8]),
        flag("loader.batch_size", [8]),
        flag("model.expand", [2]),
        flag("model.layer.postact", ['glu']),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def youtube_snet_sweep_2():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.n_layers", [2]),
        lzip([
            flag("model.expand", [4, 4, 2]),
            flag("model.ff", [4, 4, 2]),
            flag("model.pool", [[4, 4], [8, 8, 2], [4, 4]]),
            flag("loader.batch_size", [2, 1, 2]),
        ]),
        flag("model.act_pool", ['glu']),
        flag("model.layer.trainable.A", [2]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/'])
    ])

    return sweep


def youtube_snet_actglu_prenorm_pool():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.sample_len", [65536]),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.n_layers", [8]),
        flag("model.ff", [2]),
        flag("model.expand", [2]),
        lzip([
            flag("model.pool", [[4, 4, 4], [4, 4], [4, 4]]),
            flag("model.layer.postact", ['null', 'glu', 'null']),
            flag("model.prenorm", [True, True, False]),
            flag("loader.batch_size", [2, 2, 2]),
        ]),
        flag("model.layer.trainable.A", [2]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/'])
    ])

    return sweep


def beethoven():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset.path", ['/home/workspace/hippo/data/beethoven/']),
        flag("dataset.sample_len", [128000]),
        flag("dataset.quantization", ['linear']),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.ff", [2]),
        flag("model.pool", [[4, 4]]),
        flag("model.n_layers", [8]),
        flag("model.layer.trainable.A", [2]),
        flag("loader.batch_size", [1]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def youtube_isotropic():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.expand", [0]),
        flag("model.ff", [4]),
        flag("model.pool", [[]]),
        flag("model.d_model", [256]),
        flag("loader.batch_size", [1]),
        lzip([
            flag("model.n_layers", [4, 8]),    
            flag("dataset.sample_len", [65536, 32768]),
        ]),
        flag("model.layer.trainable.A", [2]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),  
        flag("dataset.drop_last", [False]),
        flag("decoder.mode", ['ragged']),
    ])

    return sweep

# 

def youtube_isotropic_resume():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.expand", [0]),
        flag("model.ff", [4]),
        flag("model.pool", [[]]),
        flag("model.d_model", [256]),
        flag("loader.batch_size", [1]),
        lzip([
            flag("model.n_layers", [4, 8]),    
            flag("dataset.sample_len", [65536, 32768]),
            flag(
                "trainer.resume_from_checkpoint", 
                [
                    '/home/workspace/projects/hippo/outputs/2022-01-21/08-19-30/checkpoints/val/loss-v8.ckpt',
                    '/home/workspace/projects/hippo/outputs/2022-01-21/22-53-34/checkpoints/val/loss-v6.ckpt',
                ]
            )
        ]),
        flag("model.layer.trainable.A", [2]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),  
        flag("dataset.drop_last", [False]),
        flag("decoder.mode", ['ragged']),
    ])

    return sweep

def youtube_isotropic_resume_2():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.expand", [0]),
        flag("model.ff", [4]),
        flag("model.pool", [[]]),
        flag("model.d_model", [256]),
        flag("loader.batch_size", [1]),
        lzip([
            flag("model.n_layers", [4]),    
            flag("dataset.sample_len", [65536]),
            flag(
                "trainer.resume_from_checkpoint", 
                [
                    '/home/workspace/projects/hippo/outputs/2022-01-22/20-14-07/checkpoints/val/loss-v2.ckpt',
                ]
            )
        ]),
        flag("model.layer.trainable.A", [2]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),  
        flag("dataset.drop_last", [False]),
        flag("decoder.mode", ['ragged']),
    ])

    return sweep

def beethoven_shorter():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/beethoven/']),
        lzip([
            flag("dataset.sample_len", [64000, 32000, 16000]),
            flag("loader.batch_size", [2, 4, 8]),
        ]),
        flag("dataset.quantization", ['linear']),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.ff", [2]),
        flag("model.pool", [[4, 4]]),
        flag("model.n_layers", [8]),
        flag("model.layer.trainable.A", [2]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep


def beethoven_shorter_resume():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/beethoven/']),
        lzip([
            flag("dataset.sample_len", [16000]),
            flag("loader.batch_size", [8]),
            flag(
                "trainer.resume_from_checkpoint", 
                [
                    '/home/workspace/projects/hippo/outputs/2022-01-22/20-19-30/checkpoints/val/loss-v9.ckpt',
                ]
            )
        ]),
        flag("dataset.quantization", ['linear']),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.ff", [2]),
        flag("model.pool", [[4, 4]]),
        flag("model.n_layers", [8]),
        flag("model.layer.trainable.A", [2]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def beethoven_shorter_A_1():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/beethoven/']),
        lzip([
            flag("dataset.sample_len", [16000]),
            flag("loader.batch_size", [8]),
        ]),
        flag("dataset.quantization", ['linear']),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.ff", [2]),
        flag("model.pool", [[4, 4]]),
        flag("model.n_layers", [8]),
        flag("model.layer.trainable.A", [1]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def beethoven_shorter_all_A_1():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/beethoven/']),
        lzip([
            flag("dataset.sample_len", [64000, 32000]),
            flag("loader.batch_size", [2, 4]),
        ]),
        flag("dataset.quantization", ['linear']),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.ff", [2]),
        flag("model.pool", [[4, 4]]),
        flag("model.n_layers", [8]),
        flag("model.layer.trainable.A", [1]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def youtube_ablationssm():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        lzip([
            flag("model.layer.trainable.A", [1, 1, 0, 1, 1, 0]),
            flag("model.layer.trainable.B", [0, 1, 0, 0, 0, 1]),
            flag("model.layer.trainable.P", [1, 1, 0, 0, 1, 0]),
            flag("model.layer.trainable.Q", [1, 1, 0, 0, 1, 0]),
            flag("+model.layer.tied_lr", [True, False, False, False, False, False]),
        ]),
        flag("model.layer.trainable.C", [1]),
        flag("model.layer.trainable.dt", [0]),
        flag("model.expand", [2]),
        flag("model.ff", [2]),
        flag("model.n_layers", [2]),
        flag("loader.batch_size", [4]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [20]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep

def youtube_ablationssm_2():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        lzip([
            flag("model.layer.trainable.A", [1]),
            flag("model.layer.trainable.B", [1]),
            flag("model.layer.trainable.P", [1]),
            flag("model.layer.trainable.Q", [1]),
            flag("+model.layer.tied_lr", [True]),
        ]),
        flag("model.layer.trainable.C", [1]),
        flag("model.layer.trainable.dt", [0]),
        flag("model.expand", [2]),
        flag("model.ff", [2]),
        flag("model.n_layers", [2]),
        flag("loader.batch_size", [4]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [20]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep


def beethoven_8s_A_1():
    """Run this later -- doesn't run on a V100."""
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/beethoven/']),
        lzip([
            flag("dataset.sample_len", [128000]),
            flag("loader.batch_size", [1]),
            flag("model.layer.trainable.A", [1]),
            flag("model.layer.trainable.B", [1]),
            flag("model.layer.trainable.P", [1]),
            flag("model.layer.trainable.Q", [1]),
            flag("+model.layer.tied_lr", [True]),
        ]),
        flag("dataset.quantization", ['linear']),
        flag("model", ['snet']),
        flag("model.expand", [2]),
        flag("model.ff", [2]),
        flag("model.pool", [[4, 4]]),
        flag("model.n_layers", [8]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("trainer.max_epochs", [500]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),
    ])

    return sweep

def youtube_ablationssm_3():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.path", ['/home/workspace/hippo/data/youtube_mix/']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        lzip([
            flag("model.layer.trainable.A", [1]),
            flag("model.layer.trainable.B", [1]),
            flag("model.layer.trainable.P", [1]),
            flag("model.layer.trainable.Q", [1]),
            flag("+model.layer.tied_lr", [True]),
            flag("+model.layer.hurwitz", [True]),
        ]),
        flag("model.layer.trainable.C", [1]),
        flag("model.layer.trainable.dt", [0]),
        flag("model.expand", [2]),
        flag("model.ff", [2]),
        flag("model.n_layers", [2]),
        flag("loader.batch_size", [4]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [20]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
    ])

    return sweep


def youtube_ablationssm_3_resume():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.path", ['/home/workspace/hippo/data/youtube_mix/']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        lzip([
            flag("model.layer.trainable.A", [1]),
            flag("model.layer.trainable.B", [1]),
            flag("model.layer.trainable.P", [1]),
            flag("model.layer.trainable.Q", [1]),
            flag("+model.layer.tied_lr", [True]),
            flag("+model.layer.hurwitz", [True]),
        ]),
        flag("model.layer.trainable.C", [1]),
        flag("model.layer.trainable.dt", [0]),
        flag("model.expand", [2]),
        flag("model.ff", [2]),
        flag("model.n_layers", [2]),
        flag("loader.batch_size", [4]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [20]),
        flag("task.metrics", [['bpb', 'accuracy', 'accuracy@3', 'accuracy@5', 'accuracy@10']]),
        flag("trainer.resume_from_checkpoint", ['/home/workspace/hippo/outputs/2022-02-02/22-47-27/checkpoints/last.ckpt'])
    ])

    return sweep

def youtube_isotropic_new():
    sweep = prod([
        flag("experiment", ['s4-qautomusic']),
        flag("dataset", ['youtubemix']),
        flag("dataset.path", ['/home/workspace/projects/hippo/data/youtube_mix/']),
        flag("dataset.quantization", ['mu-law']),
        flag("model", ['snet']),
        flag("model.expand", [0]),
        flag("model.ff", [4]),
        flag("model.pool", [[]]),
        flag("model.d_model", [256]),
        flag("loader.batch_size", [1]),
        lzip([
            flag("model.n_layers", [4, 8]),    
            flag("dataset.sample_len", [65536, 32768]),
        ]),
        flag("model.layer.trainable.A", [True]),
        flag("model.layer.trainable.B", [True]),
        flag("model.layer.trainable.P", [True]),
        flag("model.layer.trainable.dt", [True]),
        flag("model.layer.postact", ['glu']),
        flag("model.layer.hurwitz", [True]),
        flag("model.layer.tie_state", [True]),
        flag("trainer.max_epochs", [1000]),
        flag("optimizer.lr", [0.004]),
        flag("scheduler.patience", [20]),
        flag("callbacks.model_checkpoint.save_top_k", [10]),  
        flag("dataset.drop_last", [False]),
        flag("decoder.mode", ['ragged']),
    ])

    return sweep

def youtube_statespaces_repro(): # to run
    sweep = prod([
        flag("experiment", ['sashimi-beethoven', 'sashimi-youtubemix', 'sashimi-sc09']),
    ])

    return sweep