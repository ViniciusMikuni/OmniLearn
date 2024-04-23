import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import sys
import horovod.tensorflow.keras as hvd
from omnifold import OmniFold, Classifier

import utils



hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

parser = OptionParser(usage="%prog [opt]  inputFiles")

parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET", help="Folder containing input files")
parser.add_option("--dataset", type="string", default="omnifold", help="Dataset to use")
parser.add_option("--mode", type="string", default="classifier", help="Loss type to train the model: available options are [all/classifier/generator]")
parser.add_option("--batch", type=int, default=512, help="Batch size")
parser.add_option("--epoch", type=int, default=20, help="Max epoch")
parser.add_option("--num_iter", type=int, default=6, help="OmniFold iterations")
parser.add_option("--lr", type=float, default=3e-5, help="learning rate")
parser.add_option('--fine_tune', action='store_true', default=False,help='Fine tune a model')
#Model parameters
parser.add_option('--local', action='store_true', default=False,help='Use local embedding')
parser.add_option("--num_layers", type=int, default=8, help="Number of transformer layers")
parser.add_option("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")

parser.add_option("--wd", type=float, default=0.1, help="weight decay")
parser.add_option("--b1", type=float, default=0.95, help="lion beta1")
parser.add_option("--b2", type=float, default=0.99, help="lion beta2")
parser.add_option("--lr_factor", type=float, default=5., help="factor for slower learning rate")

parser.add_option('--simple', action='store_true', default=False,help='Use simplified head model')
parser.add_option('--talking_head', action='store_true', default=False,help='Use talking head attention instead of standard attention')
parser.add_option('--layer_scale', action='store_true', default=False,help='Use layer scale in the residual connections')


(flags, args) = parser.parse_args()

    
mc = utils.OmniDataLoader(os.path.join(flags.folder,'OmniFold','train_pythia.h5'),hvd.rank(),hvd.size())
data = utils.OmniDataLoader(os.path.join(flags.folder,'OmniFold','train_herwig.h5'),hvd.rank(),hvd.size())

if flags.fine_tune:
    model_name = utils.get_model_name(flags,flags.fine_tune).replace(flags.dataset,'jetclass').replace('fine_tune','baseline').replace(flags.mode,'all')
    model_name = os.path.join(flags.folder,'checkpoints',model_name)
else:
    model_name = None
        

model1 = Classifier(num_feat=mc.num_feat,
                    num_jet=mc.num_jet,
                    num_classes=mc.num_classes,
                    local = flags.local,
                    num_layers = flags.num_layers, 
                    drop_probability = flags.drop_probability,
                    simple = flags.simple, layer_scale = flags.layer_scale,
                    talking_head = flags.talking_head,
                    mode = 'classifier',
                    class_activation=None,
                    fine_tune = flags.fine_tune,
                    model_name = model_name,
                    )

model2 = keras.models.clone_model(model1)


    
mfold = OmniFold(version = 'fine_tune' if flags.fine_tune else 'baseline',
                 num_iter = flags.num_iter,checkpoint_folder = flags.folder,
                 batch_size = flags.batch,epochs=flags.epoch,
                 wd=flags.wd,b1=flags.b1,b2=flags.b2,lr_factor=flags.lr_factor,
                 lr = flags.lr,fine_tune = flags.fine_tune, size=hvd.size())
mfold.mc = mc
mfold.data = data
mfold.Preprocessing(model1,model2)
mfold.Unfold()
