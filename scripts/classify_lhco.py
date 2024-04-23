import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import sys
import horovod.tensorflow.keras as hvd
from evaluate_lhco import get_features
from PET_lhco import  Classifier
import utils
import pickle
import gc

hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--dataset", type="string", default="lhco", help="Folder containing input files")
parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
parser.add_option("--mode", type="string", default="classifier", help="Loss type to train the model: available options are [all/classifier/generator]")
parser.add_option('--SR', action='store_true', default=False,help='Generate SR data')
parser.add_option("--batch", type=int, default=128, help="Batch size")
parser.add_option("--epoch", type=int, default=10, help="Max epoch")
parser.add_option("--steps", type=int, default=40, help="Number of steps used in the cosine learning rate")
parser.add_option("--nsig", type=int, default=1000, help="Number of signal events used in the training")
parser.add_option("--lr", type=float, default=3e-5, help="learning rate")
parser.add_option('--fine_tune', action='store_true', default=False,help='Fine tune a model')
parser.add_option('--ideal', action='store_true', default=False,help='Train idealized model')
parser.add_option("--nid", type=int, default=0, help="ID of the training for multiple runs")
#Model parameters
parser.add_option('--local', action='store_true', default=False,help='Use local embedding')
parser.add_option("--num_layers", type=int, default=8, help="Number of transformer layers")
parser.add_option("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
parser.add_option('--simple', action='store_true', default=False,help='Use simplified head model')
parser.add_option('--weighted', action='store_true', default=False,help='Weight the SR based on the SB')
parser.add_option('--talking_head', action='store_true', default=False,help='Use talking head attention instead of standard attention')
parser.add_option('--layer_scale', action='store_true', default=False,help='Use layer scale in the residual connections')


(flags, args) = parser.parse_args()
scale_lr = flags.lr*np.sqrt(hvd.size()) 

if flags.fine_tune:
    model_name = utils.get_model_name(flags,flags.fine_tune).replace(flags.dataset,'jetclass').replace('fine_tune','baseline').replace(flags.mode,'all')
    model_name = os.path.join(flags.folder,'checkpoints',model_name)
    lr_factor = 10.
else:
    model_name = None
    lr_factor = 1.


region = "SR" if flags.SR else "SB"
sample_name = utils.get_model_name(flags,flags.fine_tune).replace(".weights.h5","_{}.h5".format(region)).replace("classifier","generator")


if flags.ideal:
    assert region == 'SR', "ERROR: Only SR background samples are available"
    train = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','train_background_{}_extended.h5'.format(region)),flags.batch,hvd.rank(),hvd.size(),nevts=320000)
    test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','val_background_{}_extended.h5'.format(region)),flags.batch,hvd.rank(),hvd.size(),nevts=35555)
else:
    train = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO',"train_"+sample_name),flags.batch,hvd.rank(),hvd.size(),nevts=320000)
    test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO',"test_"+sample_name),flags.batch,hvd.rank(),hvd.size(),nevts=35555)

    
data_train = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','train_background_{}.h5'.format(region)),flags.batch,hvd.rank(),hvd.size(),nevts=90000)
data_test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','val_background_{}.h5'.format(region)),flags.batch,hvd.rank(),hvd.size(),nevts=10000)



model = Classifier(num_feat=train.num_feat,
                   num_jet=train.num_jet,
                   num_classes=train.num_classes,
                   local = flags.local,
                   num_layers = flags.num_layers, 
                   drop_probability = flags.drop_probability,
                   simple = flags.simple, layer_scale = flags.layer_scale,
                   talking_head = flags.talking_head,
                   mode = flags.mode,
                   fine_tune = flags.fine_tune,
                   model_name = model_name,
                   )

if flags.weighted:
    assert flags.SR, "ERROR: Can only reweight the SR"
    from scipy.special import expit
    model_reweight = keras.models.clone_model(model)

    model_reweight.load_weights(os.path.join(flags.folder,'checkpoints',utils.get_model_name(flags,flags.fine_tune,add_string='_SB_1000')))    
    X,_ = train.make_eval_data()
    train.w = expit(model.predict(X,verbose=hvd.rank()==0))
    train.w = np.nan_to_num(train.w / (1.0 -train.w),posinf=1.0,neginf=0.0)
    train.w *= train.w.shape[0]/np.sum(train.w)
    X,_ = test.make_eval_data()
    test.w = expit(model.predict(X,verbose=hvd.rank()==0))
    test.w = np.nan_to_num(test.w / (1.0 -test.w),posinf=1.,neginf=0.0)
    test.w *= test.w.shape[0]/np.sum(test.w)
    del model_reweight, X
    gc.collect()
else:
    train.w = np.ones((train.y.shape[0],1))
    test.w = np.ones((test.y.shape[0],1))

train.w *= data_train.nevts/train.nevts
test.w *= data_test.nevts/test.nevts
    

if flags.SR:
    signal_train = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','train_signal_{}.h5'.format(region)),flags.batch,hvd.rank(),hvd.size(),nevts = int(flags.nsig*0.9))
    train.combine([data_train,signal_train],use_weights=True)    
    signal_test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','val_signal_{}.h5'.format(region)),flags.batch,hvd.rank(),hvd.size(),nevts = int(flags.nsig*0.1))
    test.combine([signal_test,data_test],use_weights=True)
else:
    train.combine([data_train],use_weights=True)
    test.combine([data_test],use_weights=True)
      
lr_schedule_body = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=flags.lr/lr_factor,
    warmup_target = scale_lr/lr_factor,
    warmup_steps= 3*train.nevts//hvd.size()//flags.batch,
    decay_steps=flags.steps*train.nevts//hvd.size()//flags.batch,    
)

lr_schedule_head = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=flags.lr,
    warmup_target = scale_lr,
    warmup_steps= 3*train.nevts//hvd.size()//flags.batch,
    decay_steps=flags.steps*train.nevts//hvd.size()//flags.batch,
)

opt_body = keras.optimizers.Lion(
    learning_rate = lr_schedule_body,
    weight_decay=0.1,
    beta_1=0.95)

opt_body = hvd.DistributedOptimizer(opt_body)

opt_head = keras.optimizers.Lion(
    learning_rate = lr_schedule_head,
    weight_decay=0.1,
    beta_1=0.95)

opt_head = hvd.DistributedOptimizer(opt_head)
model.compile(opt_body,opt_head)



callbacks=[
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    ReduceLROnPlateau(monitor='val_loss',patience=30000, min_lr=1e-6),
]

if hvd.rank()==0:
    add_string = '_{}_{}'.format(region,flags.nsig)
    if flags.ideal:
        add_string += '_ideal'
    if flags.nid > 0:
        add_string += '_{}'.format(flags.nid)
    checkpoint = ModelCheckpoint(
        os.path.join(flags.folder,'checkpoints',utils.get_model_name(flags,flags.fine_tune,add_string=add_string)),
        #save_best_only=True,
        mode='auto',
        save_weights_only=True,
        period=1)
    callbacks.append(checkpoint)


hist =  model.fit(train.make_tfdata(classification=True),
                  epochs=flags.epoch,
                  validation_data=test.make_tfdata(classification=True),
                  batch_size=flags.batch,
                  callbacks=callbacks,                  
                  steps_per_epoch=train.steps_per_epoch,
                  validation_steps =test.steps_per_epoch,
                  verbose=hvd.rank() == 0,
)
if hvd.rank() ==0:
    with open(os.path.join(flags.folder,'histories',utils.get_model_name(flags,flags.fine_tune).replace(".weights.h5",".pkl")),"wb") as f:
        pickle.dump(hist.history, f)
