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
from omnifold import Classifier


hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def print_metrics(y_pred,y, thresholds,multi_label=False):
    if multi_label:
        print("AUC: {}".format(metrics.roc_auc_score(y, y_pred,average='macro',multi_class='ovo')))
        
        one_hot_predictions = np.zeros_like(y_pred)
        one_hot_predictions[np.arange(len(y_pred)), y_pred.argmax(axis=-1)] = 1
        
        print('Acc: {}'.format(metrics.accuracy_score(y,one_hot_predictions)))    
        
        bkg_idx = 0
        
        for idx in range(np.shape(y)[-1]):
            if idx == bkg_idx:continue
            mask = (y[:,idx]==1) | (y[:,bkg_idx]==1) #only keep signal+bkg
            pred_sb = y_pred[mask,idx]/(y_pred[mask,idx] + y_pred[mask,bkg_idx])
            fpr, tpr, _ = metrics.roc_curve(y[mask,idx], pred_sb)
            
            for threshold in thresholds:
                bineff = np.argmax(tpr>threshold)
                print('Class {} effS at {} 1.0/effB = {}'.format(idx,tpr[bineff],1.0/fpr[bineff]))

    else:
        print("AUC: {}".format(metrics.roc_auc_score(y, y_pred)))
        print('Acc: {}'.format(metrics.accuracy_score(y,y_pred>0.5)))
        fpr, tpr, _ = metrics.roc_curve(y, y_pred)

        for threshold in thresholds:
            bineff = np.argmax(tpr>threshold)
            print('effS at {} 1.0/effB = {}'.format(tpr[bineff],1.0/fpr[bineff]))

        
        tpr=tpr[fpr>1e-4]
        fpr=fpr[fpr>1e-4]
        sic = np.ma.divide(tpr,np.sqrt(fpr)).filled(0)
        print("Max SIC: {}".format(np.max(sic)))


    
if __name__=='__main__':
        
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--dataset", type="string", default="top", help="Folder containing input files")
    parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_option("--batch", type=int, default=5000, help="Batch size")
    parser.add_option('--load', action='store_true', default=False, help='Load pre-evaluated npy files')
    parser.add_option("--mode", type="string", default="classifier", help="Loss type to train the model: available options are [all/classifier/generator]")
    parser.add_option('--fine_tune', action='store_true', default=False,help='Fine tune a model')
    parser.add_option("--nid", type=int, default=0, help="Training ID for multiple trainings")
    #Model parameters
    parser.add_option('--local', action='store_true', default=False,help='Use local embedding')
    parser.add_option("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_option("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_option('--simple', action='store_true', default=False,help='Use average instead of class token')
    parser.add_option('--talking_head', action='store_true', default=False,help='Use talking head attention instead of standard attention')
    parser.add_option('--layer_scale', action='store_true', default=False,help='Use layer scale in the residual connections')


    (flags, args) = parser.parse_args()

    multi_label = True
    if flags.dataset == 'top':
        test = utils.TopDataLoader(os.path.join(flags.folder,'TOP', 'test_ttbar.h5'),flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.3, 0.5]
        folder_name = 'TOP'

    elif flags.dataset == 'qg':
        test = utils.QGDataLoader(os.path.join(flags.folder,'QG', 'test_qg.h5'),flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.3,0.5]
        folder_name = 'QG'

    elif flags.dataset == 'atlas':
        test = utils.AtlasDataLoader(os.path.join(flags.folder,'ATLASTOP', 'test_atlas.h5'),flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.5,0.8]
        folder_name = 'ATLASTOP'
        multi_label = False
    elif flags.dataset == 'atlas_small':
        test = utils.AtlasDataLoader(os.path.join(flags.folder,'ATLASTOP', 'test_atlas.h5'),flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.5,0.8]
        folder_name = 'ATLASTOP'
        multi_label = False
    elif flags.dataset == 'h1':
        test = utils.H1DataLoader(os.path.join(flags.folder,'H1', 'test.h5'),flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.5,0.1]
        folder_name = 'H1'

    elif flags.dataset == 'cms':
        test = utils.CMSQGDataLoader(os.path.join(flags.folder,'CMSQG', 'test_qgcms_pid.h5'),flags.batch,rank = hvd.rank(),size = hvd.size())
        threshold = [0.5,0.8]
        folder_name = 'CMSQG'
    elif flags.dataset == 'jetclass':
        test = utils.JetClassDataLoader(os.path.join(flags.folder,'JetClass','val',rank = hvd.rank(),size = hvd.size()),
                                        flags.batch)
        threshold = [0.5]
        folder_name = 'JetClass/test'

    if flags.load:
        print("Loading saved npy files")
        npy_file = os.path.join(flags.folder,folder_name,'npy','{}.npy'.format(
            utils.get_model_name(flags,fine_tune=flags.fine_tune,
                                 add_string = '_{}'.format(flags.nid) if flags.nid > 0 else '').replace('.h5','')))
        
        data = np.load(npy_file,allow_pickle=True)
        y = data.item()['y']
        pred = data.item()['pred']
    else:
        X,y = test.make_eval_data()
        if 'atlas' in flags.dataset :
            model_function = Classifier
            activation = 'sigmoid'
        else:
            model_function = PET
            activation = 'softmax'
        model = model_function(num_feat=test.num_feat,
                               num_jet=test.num_jet,
                               num_classes=test.num_classes,
                               local = flags.local,
                               num_layers = flags.num_layers, 
                               drop_probability = flags.drop_probability,
                               simple = flags.simple, layer_scale = flags.layer_scale,
                               talking_head = flags.talking_head,
                               mode = flags.mode,
                               class_activation=activation
                               )


        if flags.nid>0:
            #Load alternative runs
            add_string = '_{}'.format(flags.nid)
        else:
            add_string = ''
        
        model.load_weights(
            os.path.join(flags.folder,'checkpoints',
                         utils.get_model_name(flags,fine_tune=flags.fine_tune,add_string=add_string)))
        pred = model.predict(X,verbose=hvd.rank()==0)[0]

        if not os.path.exists(os.path.join(flags.folder,folder_name,'npy')):
            os.makedirs(os.path.join(flags.folder,folder_name,'npy'))
    
        np.save(os.path.join(flags.folder,folder_name,'npy',
                             '{}.npy'.format(utils.get_model_name(flags,fine_tune=flags.fine_tune,
                                                                  add_string = '_{}'.format(flags.nid) if flags.nid > 0 else '').replace('.h5',''))),
                {'y':hvd.allgather(tf.constant(y)).numpy(),
                 'pred':hvd.allgather(tf.constant(pred)).numpy()})
        

    #Starting the evaluation of the results        
    print_metrics(pred,y,threshold,multi_label)
