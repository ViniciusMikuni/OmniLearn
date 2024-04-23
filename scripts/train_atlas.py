import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import sys
import horovod.tensorflow.keras as hvd
from PET import PET
import utils
import pickle

from omnifold import Classifier

hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--dataset", type="string", default="atlas", help="Dataset tp use")
parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
parser.add_option("--mode", type="string", default="classifier", help="Loss type to train the model: available options are [all/classifier/generator]")
#parser.add_option("--folder", type="string", default="/global/cfs/cdirs/m3929/", help="Folder containing input files")
parser.add_option("--batch", type=int, default=256, help="Batch size")
parser.add_option("--epoch", type=int, default=40, help="Max epoch")
parser.add_option("--warm_epoch", type=int, default=1, help="Warm up epochs")
parser.add_option("--reduce_epoch", type=int, default=1, help="Epochs before reducing lr")
parser.add_option("--stop_epoch", type=int, default=5, help="Epochs before reducing lr")
parser.add_option("--lr", type=float, default=3e-5, help="learning rate")
parser.add_option('--fine_tune', action='store_true', default=False,help='Fine tune a model')

#Model parameters
parser.add_option('--local', action='store_true', default=False,help='Use local embedding')
parser.add_option("--num_layers", type=int, default=8, help="Number of transformer layers")
parser.add_option("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
parser.add_option('--simple', action='store_true', default=False,help='Use simplified head model')
parser.add_option('--talking_head', action='store_true', default=False,help='Use talking head attention instead of standard attention')
parser.add_option('--layer_scale', action='store_true', default=False,help='Use layer scale in the residual connections')


(flags, args) = parser.parse_args()

scale_lr = flags.lr*np.sqrt(hvd.size()) 



train = utils.AtlasDataLoader(os.path.join(flags.folder,'ATLASTOP', 'train_atlas.h5'),flags.batch,hvd.rank(),hvd.size(),is_small='small' in flags.dataset)
test = utils.AtlasDataLoader(os.path.join(flags.folder,'ATLASTOP', 'val_atlas.h5'),flags.batch,hvd.rank(),hvd.size(),is_small='small' in flags.dataset)


if flags.fine_tune:
    model_name = utils.get_model_name(flags,flags.fine_tune).replace(flags.dataset,'jetclass').replace('fine_tune','baseline').replace(flags.mode,'all')
    if hvd.rank()==0:print("Loading model {}".format(model_name))
    model_name = os.path.join(flags.folder,'checkpoints',model_name)
    lr_factor = 5.
else:
    model_name = None
    lr_factor = 1.



model = Classifier(num_feat=train.num_feat,
                   num_jet=train.num_jet,
                   num_classes=train.num_classes,
                   local = flags.local,
                   num_layers = flags.num_layers, 
                   drop_probability = flags.drop_probability,
                   simple = flags.simple, layer_scale = flags.layer_scale,
                   talking_head = flags.talking_head,
                   mode = flags.mode,
                   class_activation= None,
                   fine_tune = flags.fine_tune,
                   model_name = model_name,
                   )


lr_schedule_body = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=flags.lr/lr_factor,
    warmup_target = scale_lr/lr_factor,
    warmup_steps= flags.warm_epoch*train.nevts//hvd.size()//flags.batch,
    decay_steps=flags.epoch*train.nevts//hvd.size()//flags.batch,
)


lr_schedule_head = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=flags.lr,
    warmup_target = scale_lr,
    warmup_steps= flags.warm_epoch*train.nevts//hvd.size()//flags.batch,
    decay_steps=flags.epoch*train.nevts//hvd.size()//flags.batch,
)

    
opt_body = keras.optimizers.Lion(
    learning_rate = lr_schedule_body,
    weight_decay=1e-4*lr_factor,
    beta_1=0.95,
    beta_2=0.99,
    clipnorm = 1.0
)
opt_body = hvd.DistributedOptimizer(opt_body)
opt_head = keras.optimizers.Lion(
    learning_rate = lr_schedule_head,
    weight_decay=1e-4,
    beta_1=0.95,
    beta_2=0.99,
    clipnorm = 1.0
)

opt_head = hvd.DistributedOptimizer(opt_head)
model.compile(opt_body,opt_head)
# model.compile(weighted_metrics=[],
#               optimizer=opt,
#               #run_eagerly=True,
#               metrics=['accuracy'],
#               experimental_run_tf_function=False,
#               )


callbacks=[
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    EarlyStopping(patience=flags.stop_epoch,restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',patience=1000, min_lr=1e-6),
    # hvd.callbacks.LearningRateWarmupCallback(initial_lr=scale_lr,
    #                                          warmup_epochs=flags.warm_epoch, verbose=1,
    #                                          steps_per_epoch = train.steps_per_epoch),
]

if hvd.rank()==0:

    checkpoint = ModelCheckpoint(
        os.path.join(flags.folder,'checkpoints',utils.get_model_name(flags,flags.fine_tune)),
        save_best_only=True,mode='auto',
        save_weights_only=True,
        period=1)
    callbacks.append(checkpoint)
    
hist =  model.fit(train.make_tfdata(),
                  epochs=flags.epoch,
                  validation_data=test.make_tfdata(),
                  batch_size=flags.batch,
                  callbacks=callbacks,                  
                  steps_per_epoch=train.steps_per_epoch,
                  validation_steps =test.steps_per_epoch,
                  verbose=hvd.rank() == 0,
)
if hvd.rank() ==0:
    with open(os.path.join(flags.folder,'histories',utils.get_model_name(flags,flags.fine_tune).replace(".weights.h5",".pkl")),"wb") as f:
        pickle.dump(hist.history, f)
                            
