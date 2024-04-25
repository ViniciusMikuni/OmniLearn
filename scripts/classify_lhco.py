import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import schedules, Lion
import horovod.tensorflow.keras as hvd
import argparse
import logging
import pickle
import gc

from evaluate_lhco import get_features
from PET_lhco import Classifier
import utils

def setup_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def configure_optimizers(flags, train_loader,lr_factor = 1.0):
    scale_lr = flags.lr * np.sqrt(hvd.size())
    lr_schedule = schedules.CosineDecay(
        initial_learning_rate=flags.lr/lr_factor,
        warmup_target=scale_lr/lr_factor,
        warmup_steps=3*train_loader.nevts//flags.batch//hvd.size(),
        decay_steps=flags.epoch*train_loader.nevts//flags.batch//hvd.size(),
    )
    optimizer = Lion(
        learning_rate=lr_schedule,
        clipnorm=1.0,
        beta_1=0.95,
    )
    return hvd.DistributedOptimizer(optimizer)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Classification script for LHCO data using PET classifier.")
    parser.add_argument("--dataset", default="lhco", help="Dataset to use")
    parser.add_argument("--folder", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_argument("--mode", default="classifier", help="Loss type to train the model")
    parser.add_argument("--SR", action="store_true", default=False, help="Generate SR data")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--epoch", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--steps", type=int, default=40, help="Number of steps in the cosine learning rate schedule")
    parser.add_argument("--nsig", type=int, default=1000, help="Number of signal events used in the training")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--lr_factor", type=float, default=10.0, help="Factor to adjust learning rate")
    parser.add_argument("--fine_tune", action="store_true", default=False, help="Fine tune a model")
    parser.add_argument("--ideal", action="store_true", default=False, help="Train idealized model")
    parser.add_argument("--nid", type=int, default=0, help="ID of the training for multiple runs")
    parser.add_argument("--local", action="store_true", default=False, help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Drop probability")
    parser.add_argument("--simple", action="store_true", default=False, help="Use simplified head model")
    
    parser.add_argument("--talking_head", action="store_true", default=False, help="Use talking head attention")
    parser.add_argument("--layer_scale", action="store_true", default=False, help="Use layer scale in the residual connections")
    args = parser.parse_args()
    return args


def get_data_loader(flags,region):
    if flags.ideal:
        assert region == 'SR', "ERROR: Only SR background samples are available for idealized training"
        train = utils.LHCODataLoader(os.path.join(flags.folder, 'LHCO', f'train_background_{region}_extended.h5'), flags.batch, hvd.rank(), hvd.size(), nevts=320000)
        test = utils.LHCODataLoader(os.path.join(flags.folder, 'LHCO', f'val_background_{region}_extended.h5'), flags.batch, hvd.rank(), hvd.size(), nevts=35555)
    else:
        sample_name = utils.get_model_name(flags, flags.fine_tune).replace(".weights.h5", f"_{region}.h5").replace("classifier", "generator")
        train = utils.LHCODataLoader(os.path.join(flags.folder, 'LHCO', f"train_{sample_name}"), flags.batch, hvd.rank(), hvd.size(), nevts=320000)
        test = utils.LHCODataLoader(os.path.join(flags.folder, 'LHCO', f"test_{sample_name}"), flags.batch, hvd.rank(), hvd.size(), nevts=35555)

    data_train = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','train_background_{}.h5'.format(region)),flags.batch,hvd.rank(),hvd.size(),nevts=90000)
    data_test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','val_background_{}.h5'.format(region)),flags.batch,hvd.rank(),hvd.size(),nevts=10000)


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

    return train, test

def main():
    hvd.init()
    setup_gpus()
    flags = parse_arguments()

    scale_lr = flags.lr * np.sqrt(hvd.size())
    region = "SR" if flags.SR else "SB"
    
    train,test = get_data_loader(flags,region)
    
    if flags.fine_tune:
        model_name = utils.get_model_name(flags,flags.fine_tune).replace(flags.dataset,'jetclass').replace('fine_tune','baseline').replace(flags.mode,'all')
        model_name = os.path.join(flags.folder,'checkpoints',model_name)
    else:
        model_name = None
        
    
    model = Classifier(
        num_feat=train.num_feat,
        num_jet=train.num_jet,
        num_classes=train.num_classes,
        local=flags.local,
        num_layers=flags.num_layers,
        drop_probability=flags.drop_probability,
        simple=flags.simple,
        layer_scale=flags.layer_scale,
        talking_head=flags.talking_head,
        mode=flags.mode,
        fine_tune=flags.fine_tune,
        model_name=model_name
    )

    optimizer_body = configure_optimizers(flags, train, lr_factor=flags.lr_factor if flags.fine_tune else 1)
    optimizer_head = configure_optimizers(flags, train)
    model.compile(optimizer_body, optimizer_head)

    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        ReduceLROnPlateau(monitor='val_loss', patience=30000, min_lr=1e-6)
    ]

    if hvd.rank() == 0:
        add_string = f"_{region}_{flags.nsig}"
        if flags.ideal:
            add_string += '_ideal'
        if flags.nid > 0:
            add_string += f'_{flags.nid}'
        checkpoint = ModelCheckpoint(
            os.path.join(flags.folder, 'checkpoints', utils.get_model_name(flags, flags.fine_tune, add_string=add_string)),
            save_weights_only=True,
            period=1
        )
        callbacks.append(checkpoint)

    hist = model.fit(
        train.make_tfdata(classification=True),
        epochs=flags.epoch,
        validation_data=test.make_tfdata(classification=True),
        batch_size=flags.batch,
        callbacks=callbacks,
        steps_per_epoch=train.steps_per_epoch,
        validation_steps=test.steps_per_epoch,
        verbose=hvd.rank() == 0
    )

    if hvd.rank() == 0:
        with open(os.path.join(flags.folder, 'histories', utils.get_model_name(flags, flags.fine_tune).replace(".weights.h5", ".pkl")), "wb") as f:
            pickle.dump(hist.history, f)

if __name__ == "__main__":
    main()

