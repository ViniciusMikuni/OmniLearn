import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import sys
import horovod.tensorflow.keras as hvd
from PET import PET,reset_weights
import utils
import pickle


hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--dataset", type="string", default="jetclass", help="Dataset tp use")
parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
parser.add_option("--mode", type="string", default="all", help="Loss type to train the model: available options are [all/classifier/generator]")
#parser.add_option("--folder", type="string", default="/global/cfs/cdirs/m3929/", help="Folder containing input files")
parser.add_option("--batch", type=int, default=250, help="Batch size")
parser.add_option("--epoch", type=int, default=200, help="Max epoch")
parser.add_option("--warm_epoch", type=int, default=3, help="Warm up epochs")
parser.add_option("--stop_epoch", type=int, default=30, help="Epochs before reducing lr")
parser.add_option("--lr", type=float, default=3e-5, help="learning rate")
parser.add_option("--wd", type=float, default=1e-5, help="weight decay")
parser.add_option("--b1", type=float, default=0.95, help="lion beta1")
parser.add_option("--b2", type=float, default=0.99, help="lion beta2")
parser.add_option("--lr_factor", type=float, default=10., help="factor for slower learning rate")
parser.add_option("--nid", type=int, default=0., help="Training ID for multiple trainings")
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


if flags.dataset == 'top':
    train = utils.TopDataLoader(os.path.join(flags.folder,'TOP', 'train_ttbar.h5'),flags.batch,hvd.rank(),hvd.size())
    test = utils.TopDataLoader(os.path.join(flags.folder,'TOP', 'test_ttbar.h5'),flags.batch,hvd.rank(),hvd.size())
elif flags.dataset == 'qg':
    train = utils.QGDataLoader(os.path.join(flags.folder,'QG', 'train_qg.h5'),flags.batch,hvd.rank(),hvd.size())
    test = utils.QGDataLoader(os.path.join(flags.folder,'QG', 'val_qg.h5'),flags.batch,hvd.rank(),hvd.size())
elif flags.dataset == 'cms':
    train = utils.CMSQGDataLoader(os.path.join(flags.folder,'CMSQG', 'train_qgcms_pid.h5'),flags.batch,hvd.rank(),hvd.size())
    test = utils.CMSQGDataLoader(os.path.join(flags.folder,'CMSQG', 'val_qgcms_pid.h5'),flags.batch,hvd.rank(),hvd.size())
elif flags.dataset == 'h1':
    train = utils.H1DataLoader(os.path.join(flags.folder,'H1', 'train.h5'),flags.batch,hvd.rank(),hvd.size())
    test = utils.H1DataLoader(os.path.join(flags.folder,'H1', 'val.h5'),flags.batch,hvd.rank(),hvd.size())
elif flags.dataset == 'jetclass':
    train = utils.JetClassDataLoader(os.path.join(flags.folder,'JetClass','train'),
                                     flags.batch,hvd.rank(),hvd.size())
    test = utils.JetClassDataLoader(os.path.join(flags.folder,'JetClass','test'),
                                    flags.batch,hvd.rank(),hvd.size())


model = PET(num_feat=train.num_feat,
            num_jet=train.num_jet,
            num_classes=train.num_classes,
            local = flags.local,
            num_layers = flags.num_layers, 
            drop_probability = flags.drop_probability,
            simple = flags.simple, layer_scale = flags.layer_scale,
            talking_head = flags.talking_head,
            mode = flags.mode,
            )
    
if flags.fine_tune:
    if hvd.rank()==0:
        model_name = utils.get_model_name(flags,flags.fine_tune).replace(flags.dataset,'jetclass').replace('fine_tune','baseline').replace(flags.mode,'all')
        print("Loading model {}".format(model_name))
        model.load_weights(os.path.join(flags.folder,'checkpoints',model_name),
                           by_name=True,
                           skip_mismatch=True
                           )
    #Reduce LR for fine-tuning
    lr_factor = flags.lr_factor
else:
    lr_factor = 1.


lr_schedule_body = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=flags.lr/lr_factor,
    warmup_target = scale_lr/lr_factor,
    warmup_steps= flags.warm_epoch*train.nevts//flags.batch//hvd.size(),
    decay_steps=flags.epoch*train.nevts//flags.batch//hvd.size(),
    #alpha = 1e-3,
)

lr_schedule_head = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=flags.lr,
    warmup_target = scale_lr,
    warmup_steps= flags.warm_epoch*train.nevts//flags.batch//hvd.size(),
    decay_steps=flags.epoch*train.nevts//flags.batch//hvd.size(),
    #alpha = 1e-3,
)

opt_body = keras.optimizers.Lion(
    learning_rate = lr_schedule_body,
    weight_decay=flags.wd*lr_factor,
    beta_1=flags.b1,
    beta_2=flags.b2,
)
    
opt_body = hvd.DistributedOptimizer(opt_body)

opt_heads = keras.optimizers.Lion(
    learning_rate = lr_schedule_head,
    weight_decay=flags.wd,
    beta_1=flags.b1,
    beta_2=flags.b2,
)

opt_heads = hvd.DistributedOptimizer(opt_heads)

#Allow different optimizers for fine-tuning
model.compile(opt_body,opt_heads)

callbacks=[
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    EarlyStopping(patience=flags.stop_epoch,restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',patience=200, min_lr=1e-6), #The cosine schedule already controls the LR, mostly used to print the LR value during training
]

if hvd.rank()==0:
    checkpoint = ModelCheckpoint(
        os.path.join(flags.folder,'checkpoints',utils.get_model_name(flags,flags.fine_tune,add_string="_{}".format(flags.nid) if flags.nid>0 else '')),
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
                            
