import os
import sys
import numpy as np
import h5py as h5
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
from optparse import OptionParser
from tqdm import tqdm

import horovod.tensorflow.keras as hvd
from PET import PET
import utils
from PET_lhco import Classifier

def parse_options():
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--dataset", type="string", default="lhco", help="Folder containing input files")
    parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_option("--batch", type=int, default=5000, help="Batch size")
    parser.add_option('--load', action='store_true', default=False, help='Load pre-evaluated npy files')
    parser.add_option("--mode", type="string", default="classifier", help="Loss type to train the model: available options are [all/classifier/generator]")
    parser.add_option('--fine_tune', action='store_true', default=False, help='Fine tune a model')
    parser.add_option("--nid", type=int, default=0, help="Training ID for multiple trainings")
    parser.add_option("--nsig", type=int, default=1000, help="Injected signal events for LHCO dataset")
    # Model parameters
    parser.add_option('--local', action='store_true', default=False, help='Use local embedding')
    parser.add_option("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_option("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_option('--simple', action='store_true', default=False, help='Use average instead of class token')
    parser.add_option('--ideal', action='store_true', default=False, help='Load ideal training for LHCO')
    parser.add_option('--talking_head', action='store_true', default=False, help='Use talking head attention instead of standard attention')
    parser.add_option('--layer_scale', action='store_true', default=False, help='Use layer scale in the residual connections')
    return parser.parse_args()

def load_data(flags):
    test = utils.LHCODataLoader(os.path.join(flags.folder, 'LHCO', 'val_background_SR_extended.h5'), flags.batch, hvd.rank(), hvd.size())
    sig_test = utils.LHCODataLoader(os.path.join(flags.folder, 'LHCO', 'train_signal_SR.h5'), flags.batch, hvd.rank(), hvd.size())
    test.combine([sig_test])
    return test

def evaluate_model(flags, test):
    threshold = [0.5]
    folder_name = 'LHCO'
    if flags.load:
        evaluate_existing_results(flags, folder_name, threshold)
    else:
        generate_and_save_results(flags, test, folder_name)

def print_metrics(y_pred,y, thresholds,multi_label=False):
    auc = metrics.roc_auc_score(y, y_pred)
    # print("AUC: {}".format(auc))
    # print('Acc: {}'.format(metrics.accuracy_score(y,y_pred>0.5)))
    fpr, tpr, _ = metrics.roc_curve(y, y_pred)
        
    tpr=tpr[fpr>1e-4]
    fpr=fpr[fpr>1e-4]
    sic = np.ma.divide(tpr,np.sqrt(fpr)).filled(0)
    print("Max SIC: {}".format(np.max(sic)))
    return np.max(sic), auc


def evaluate_existing_results(flags, folder_name, threshold):
    signal_values = [500, 600, 700, 800, 900, 1000, 2000, 5000, 10000]
    max_nid = 10
    sic, aucs = np.zeros((len(signal_values), max_nid)), np.zeros((len(signal_values), max_nid))
    for isig, nsig in tqdm(enumerate(signal_values), total=len(signal_values), desc='Processing Signals'):
        for nid in range(max_nid):
            print(nid)
            add_string = f'_SR_{nsig}' + ('_ideal' if flags.ideal else '') + (f'_{nid}' if nid > 0 else '')
            npy_file = os.path.join(flags.folder, folder_name, 'npy', f'{utils.get_model_name(flags, fine_tune=flags.fine_tune, add_string=add_string)}'.replace('.h5','.npy'))
            data = np.load(npy_file, allow_pickle=True)
            sic[isig, nid], aucs[isig, nid] = print_metrics(data.item()['pred'],
                                                            data.item()['y'], threshold)

    display_statistics(np.sort(sic,-1), np.sort(aucs,-1))

def generate_and_save_results(flags, test, folder_name):
    add_string = f'_SR_{flags.nsig}' + ('_ideal' if flags.ideal else '') + (f'_{flags.nid}' if flags.nid > 0 else '')
    model_name = utils.get_model_name(flags, fine_tune=flags.fine_tune, add_string=add_string)
    
    X, y = test.make_eval_data()
    model = Classifier(num_feat=test.num_feat, num_jet=test.num_jet, num_classes=test.num_classes,
                       local=flags.local, num_layers=flags.num_layers,
                       drop_probability=flags.drop_probability,
                       simple=flags.simple, layer_scale=flags.layer_scale,
                       talking_head=flags.talking_head,
                       mode=flags.mode, class_activation='sigmoid')

    model_path = os.path.join(flags.folder, 'checkpoints', model_name)
    model.load_weights(model_path)
    pred = model.predict(X, verbose=hvd.rank() == 0)[0]
    save_results(flags, folder_name, model_name.replace('.h5','.npy'), y, pred)

def save_results(flags, folder_name, model_name, y, pred):
    npy_dir = os.path.join(flags.folder, folder_name, 'npy')
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    np.save(os.path.join(npy_dir, model_name),
            {'y': hvd.allgather(tf.constant(y)).numpy(),
             'pred': hvd.allgather(tf.constant(pred)).numpy()})

def display_statistics(sic, aucs):
    sic_median, sic_low, sic_high = np.median(sic, -1), np.quantile(sic, 0.16, axis=-1), np.quantile(sic, 0.84, axis=-1)
    auc_median, auc_low, auc_high = np.median(aucs, -1), np.quantile(aucs, 0.16, axis=-1), np.quantile(aucs, 0.84, axis=-1)
    print(f"sic = {np.array2string(sic_median, separator=', ')}")
    print(f"sic_lower = {np.array2string(sic_low, separator=', ')}")
    print(f"sic_higher = {np.array2string(sic_high, separator=', ')}")
    print(f"auc = {np.array2string(auc_median, separator=', ')}")
    print(f"auc_lower = {np.array2string(auc_low, separator=', ')}")
    print(f"auc_higher = {np.array2string(auc_high, separator=', ')}")

def main():
    utils.setup_gpus()
    flags, args = parse_options()
    test = load_data(flags)
    evaluate_model(flags, test)

if __name__ == '__main__':
    main()
