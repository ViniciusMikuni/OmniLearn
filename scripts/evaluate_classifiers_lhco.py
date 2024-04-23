import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics

import horovod.tensorflow.keras as hvd
from tensorflow.keras.models import Model
import sys, gc
from PET import PET
import utils
from FPCD_lhco import Classifier 

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def print_metrics(y_pred,y, thresholds,multi_label=False):
    auc = metrics.roc_auc_score(y, y_pred)
    print("AUC: {}".format(auc))
    print('Acc: {}'.format(metrics.accuracy_score(y,y_pred>0.5)))
    fpr, tpr, _ = metrics.roc_curve(y, y_pred)
        
    tpr=tpr[fpr>1e-4]
    fpr=fpr[fpr>1e-4]
    sic = np.ma.divide(tpr,np.sqrt(fpr)).filled(0)
    print("Max SIC: {}".format(np.max(sic)))
    return np.max(sic), auc

    
if __name__=='__main__':
        
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--dataset", type="string", default="lhco", help="Folder containing input files")
    parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_option("--batch", type=int, default=5000, help="Batch size")
    parser.add_option('--load', action='store_true', default=False, help='Load pre-evaluated npy files')
    parser.add_option("--mode", type="string", default="classifier", help="Loss type to train the model: available options are [all/classifier/generator]")
    parser.add_option('--fine_tune', action='store_true', default=False,help='Fine tune a model')
    parser.add_option("--nid", type=int, default=0, help="Training ID for multiple trainings")
    parser.add_option("--nsig", type=int, default=1000, help="Injected signal events for LHCO dataset")
    #Model parameters
    parser.add_option('--local', action='store_true', default=False,help='Use local embedding')
    parser.add_option("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_option("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_option('--simple', action='store_true', default=False,help='Use average instead of class token')
    parser.add_option('--ideal', action='store_true', default=False,help='Load ideal training for LHCO')
    parser.add_option('--talking_head', action='store_true', default=False,help='Use talking head attention instead of standard attention')
    parser.add_option('--layer_scale', action='store_true', default=False,help='Use layer scale in the residual connections')


    (flags, args) = parser.parse_args()

    test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','val_background_SR_extended.h5'),flags.batch,hvd.rank(),hvd.size())
    sig_test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','train_signal_SR.h5'),flags.batch,hvd.rank(),hvd.size())
    
    test.combine([sig_test])
    threshold = [0.5]
    folder_name = 'LHCO'

        
    if flags.load:
        max_nid = 10
        print("Loading saved npy files")
        #signal_values = [500,600,700]
        signal_values = [500,600,700, 800, 900, 1000, 2000, 5000, 10000]
        sic = np.zeros((len(signal_values),max_nid))
        aucs = np.zeros((len(signal_values),max_nid))

        
        for isig,nsig in enumerate(signal_values):
            for nid in range(max_nid):
                add_string = '_SR_{}'.format(nsig)
                if flags.ideal: add_string+='_ideal'    
                if nid>0:
                    #Load alternative runs
                    add_string += '_{}'.format(nid)
                    
                npy_file = os.path.join(flags.folder,folder_name,'npy','{}.npy'.format(
                    utils.get_model_name(flags,fine_tune=flags.fine_tune,
                                         add_string = add_string).replace('.h5','')))
                print(npy_file)
                data = np.load(npy_file,allow_pickle=True)
                y = data.item()['y']
                pred = data.item()['pred']
                #Starting the evaluation of the results        
                max_sic, auc = print_metrics(pred,y,threshold)
                sic[isig,nid] = max_sic
                aucs[isig,nid] = auc


        sic_median = np.median(sic,-1)
        sic_low = np.quantile(sic,0.16,axis=-1)
        sic_high = np.quantile(sic,0.84,axis=-1)

        auc_median = np.median(aucs,-1)
        auc_low = np.quantile(aucs,0.16,axis=-1)
        auc_high = np.quantile(aucs,0.84,axis=-1)

        #print("SIC\n")
        print("sic =",np.array2string(sic_median, separator=', '))
        print("sic_lower =",np.array2string(sic_low, separator=', '))
        print("sic_higher =",np.array2string(sic_high, separator=', '))
        #print("AUC\n")
        print("auc =",np.array2string(auc_median, separator=', '))
        print("auc_lower =",np.array2string(auc_low, separator=', '))
        print("auc_higher =",np.array2string(auc_high, separator=', '))


    else:
        add_string = '_SR_{}'.format(flags.nsig)
        if flags.ideal: add_string+='_ideal'
    
        if flags.nid>0:
            #Load alternative runs
            add_string += '_{}'.format(flags.nid)

        X,y = test.make_eval_data()
        model = Classifier(num_feat=test.num_feat,
                           num_jet=test.num_jet,
                           num_classes=test.num_classes,
                           local = flags.local,
                           num_layers = flags.num_layers, 
                           drop_probability = flags.drop_probability,
                           simple = flags.simple, layer_scale = flags.layer_scale,
                           talking_head = flags.talking_head,
                           mode = flags.mode,
                           class_activation='sigmoid'
                           )



        
        model.load_weights(
            os.path.join(flags.folder,'checkpoints',
                         utils.get_model_name(flags,fine_tune=flags.fine_tune,add_string=add_string)))
        pred = model.predict(X,verbose=hvd.rank()==0)[0]

        if not os.path.exists(os.path.join(flags.folder,folder_name,'npy')):
            os.makedirs(os.path.join(flags.folder,folder_name,'npy'))
    
        np.save(os.path.join(flags.folder,folder_name,'npy',
                             '{}.npy'.format(utils.get_model_name(flags,fine_tune=flags.fine_tune,
                                                                  add_string = add_string).replace('.h5',''))),
                {'y':hvd.allgather(tf.constant(y)).numpy(),
                 'pred':hvd.allgather(tf.constant(pred)).numpy()})
        

