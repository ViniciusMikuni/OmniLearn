import numpy as np
import os
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import argparse
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from omnifold import OmniFold, Classifier
import utils


def parse_arguments():
    parser = argparse.ArgumentParser(description="OmniFold training script with distributed computing.")
    parser.add_argument("--folder", type=str, default="/pscratch/sd/v/vmikuni/PET", help="Folder containing input files")
    parser.add_argument("--dataset", type=str, default="omnifold", help="Dataset to use")
    parser.add_argument("--mode", type=str, default="classifier", help="Loss type to train the model: available options are [all/classifier/generator]")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--epoch", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("--num_iter", type=int, default=6, help="OmniFold iterations")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--fine_tune", action='store_true', default=False, help='Fine tune a model')
    parser.add_argument("--local", action='store_true', default=False, help='Use local embedding')
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--b1", type=float, default=0.95, help="Beta1 for Lion optimizer")
    parser.add_argument("--b2", type=float, default=0.99, help="Beta2 for Lion optimizer")
    parser.add_argument("--lr_factor", type=float, default=5., help="Factor for slower learning rate when fine-tuning")
    parser.add_argument("--simple", action='store_true', default=False, help='Use simplified head model')
    parser.add_argument("--talking_head", action='store_true', default=False, help='Use talking head attention')
    parser.add_argument("--layer_scale", action='store_true', default=False, help='Use layer scale in the residual connections')
    args = parser.parse_args()
    return args

def main():
    utils.setup_gpus()
    flags = parse_arguments()

    mc = utils.OmniDataLoader(os.path.join(flags.folder, 'OmniFold', 'train_pythia.h5'),flags.batch, hvd.rank(), hvd.size())
    data = utils.OmniDataLoader(os.path.join(flags.folder, 'OmniFold', 'train_herwig.h5'),flags.batch, hvd.rank(), hvd.size())

    model_name = None
    if flags.fine_tune:
        model_name = utils.get_model_name(flags, flags.fine_tune).replace(flags.dataset, 'jetclass').replace('fine_tune', 'baseline').replace(flags.mode, 'all')
        model_name = os.path.join(flags.folder, 'checkpoints', model_name)
    
    model1 = Classifier(
        num_feat=mc.num_feat,
        num_jet=mc.num_jet,
        num_classes=mc.num_classes,
        local=flags.local,
        num_layers=flags.num_layers,
        drop_probability=flags.drop_probability,
        simple=flags.simple,
        layer_scale=flags.layer_scale,
        talking_head=flags.talking_head,
        mode='classifier',
        class_activation=None,
        fine_tune=flags.fine_tune,
        model_name=model_name,
    )

    model2 = keras.models.clone_model(model1)

    mfold = OmniFold(
        version='fine_tune' if flags.fine_tune else 'baseline',
        num_iter=flags.num_iter,
        checkpoint_folder=flags.folder,
        batch_size=flags.batch,
        epochs=flags.epoch,
        wd=flags.wd,
        b1=flags.b1,
        b2=flags.b2,
        learning_rate_factor=flags.lr_factor,
        learning_rate=flags.lr,
        fine_tune=flags.fine_tune,
        size=hvd.size()
    )
    mfold.mc = mc
    mfold.data = data
    mfold.Preprocessing(model1, model2)
    mfold.Unfold()

if __name__ == "__main__":
    main()

