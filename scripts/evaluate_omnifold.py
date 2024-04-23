import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from optparse import OptionParser
import os,gc
from omnifold import OmniFold, Classifier
import tensorflow as tf
import utils, plot_utils
import horovod.tensorflow.keras as hvd
plot_utils.SetStyle()


hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


name = ["Jet Mass [GeV]","Jet Width", "$n_{constituents}$",r"$ln\rho$","$z_g$",r"$\tau_{21}$"]

def get_triangle_distance(feed_dict,weights,binning,
                          reference_name='herwig',
                          alternative_name = 'pythia_unfolded',
                          ntrials = 100):
    assert reference_name in feed_dict.keys(), "ERROR: Wrong Reference Sample"
    assert alternative_name in feed_dict.keys(), "ERROR: Wrong Alternative Sample"

    w = np.abs(binning[1] - binning[0])
    x,_ = np.histogram(feed_dict[reference_name],weights=weights[reference_name],bins=binning)
    x_norm = np.sum(x) #Assuming equidistant binning
    y,_ = np.histogram(feed_dict[alternative_name],weights=weights[alternative_name],bins=binning)
    y_norm = np.sum(y) #Assuming equidistant binning
    
    dist = 0
    for ib in range(len(x)):
        dist+=0.5*(x[ib]/x_norm - y[ib]/y_norm)**2/(x[ib]/x_norm + y[ib]/y_norm) if x[ib]/x_norm + y[ib]/y_norm >0 else 0.0

    x_plus = x + np.sqrt(x)
    x_minus = x - np.sqrt(x)
    y_plus = y + np.sqrt(y)
    y_minus = y - np.sqrt(y)
    
    results = []
    for trial in range(ntrials):
        x_ = np.random.uniform(low=x_minus, high=x_plus)
        y_ = np.random.uniform(low=y_minus, high=y_plus)
        d_ = 0.0
        for ib in range(len(x)):
            d_+=0.5*(x_[ib]/x_norm - y_[ib]/y_norm)**2/(x_[ib]/x_norm + y_[ib]/y_norm) if x_[ib]/x_norm + y_[ib]/y_norm >0 else 0.0
            results.append(d_)
        
    return dist*1e3, np.std(results)*1e3

if __name__=='__main__':

    parser = OptionParser(usage="%prog [opt]  inputFiles")
    
    parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET", help="Folder containing input files")
    parser.add_option("--dataset", type="string", default="omnifold", help="Dataset to use")
    parser.add_option("--mode", type="string", default="classifier", help="Loss type to train the model: available options are [all/classifier/generator]")
    parser.add_option('--plot_folder', default='../plots', help='Folder to store plots')
    parser.add_option("--num_iter", type=int, default=5, help="Iteratino to load")
    parser.add_option("--num_bins", type=int, default=50, help="Number of bins per histogram")

    parser.add_option('--reco', action='store_true', default=False,help='Plot reco level  results')
    
    #Model parameters
    parser.add_option('--local', action='store_true', default=False,help='Use local embedding')
    parser.add_option("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_option("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    
    parser.add_option('--simple', action='store_true', default=False,help='Use simplified head model')
    parser.add_option('--talking_head', action='store_true', default=False,help='Use talking head attention instead of standard attention')
    parser.add_option('--layer_scale', action='store_true', default=False,help='Use layer scale in the residual connections')
    
    (flags, args) = parser.parse_args()

    mc = utils.OmniDataLoader(os.path.join(flags.folder,'OmniFold','train_pythia.h5'),hvd.rank(),hvd.size())
    data = utils.OmniDataLoader(os.path.join(flags.folder,'OmniFold','train_herwig.h5'),hvd.rank(),hvd.size())
    nbins = 50
    
    binning = [
        np.linspace(0,75,nbins),
        np.linspace(0,0.5,nbins),
        np.linspace(0,80,80),
        np.linspace(-14,-2,nbins),
        np.linspace(0.0,0.6,nbins),
        np.linspace(0.0,1.2,nbins),
    ]
    
    
    if flags.reco:
        mc_inputs = mc.reco
        mc_features = mc.high_level_reco
        data_features = data.high_level_reco
    else:
        mc_inputs = mc.gen
        mc_features = mc.high_level_gen
        data_features = data.high_level_gen

    mfold = OmniFold(version = 'baseline',
                     num_iter = flags.num_iter,checkpoint_folder = flags.folder)

    model = Classifier(num_feat=mc.num_feat,
                       num_jet=mc.num_jet,
                       num_classes=mc.num_classes,
                       local = flags.local,
                       num_layers = flags.num_layers, 
                       drop_probability = flags.drop_probability,
                       simple = flags.simple, layer_scale = flags.layer_scale,
                       talking_head = flags.talking_head,
                       mode = 'classifier',
                       class_activation=None,
                       )


    

    model_name = '{}/checkpoints/OmniFold_{}_iter{}_step{}.weights.h5'.format(
        flags.folder,'baseline',0 if flags.reco else flags.num_iter,1 if flags.reco else 2)
    if hvd.rank()==0:print("Loading model {}".format(model_name))
    model.load_weights('{}'.format(model_name))
            
    unfolded_weights_baseline = hvd.allgather(mfold.reweight(mc_inputs,model,batch_size=500)).numpy()

    model_name = '{}/checkpoints/OmniFold_{}_iter{}_step{}.weights.h5'.format(
        flags.folder,'fine_tune',flags.num_iter,1 if flags.reco else 2)
    if hvd.rank()==0:print("Loading model {}".format(model_name))
    model.load_weights('{}'.format(model_name))

    unfolded_weights_fine_tune = hvd.allgather(mfold.reweight(mc_inputs,model,batch_size=500)).numpy()
    
    mc_features = hvd.allgather(mc_features).numpy()
    data_features = hvd.allgather(tf.constant(data_features)).numpy()
    
        
    if hvd.rank()==0:
        unfolded_name = 'pythia_reweighted' if flags.reco else 'pythia_unfolded'
        print(mc_features.shape[-1])
        #Event level observables
        for feature in range(mc_features.shape[-1]):
            feed_dict = {
                unfolded_name+'_fine_tune': mc_features[:,feature],
                unfolded_name+'_baseline': mc_features[:,feature],
                'pythia': mc_features[:,feature],
                'herwig': data_features[:,feature],
            }
            weights = {
                unfolded_name+'_fine_tune': unfolded_weights_fine_tune,
                unfolded_name+'_baseline': unfolded_weights_baseline,
                'pythia': np.ones(mc_features.shape[0]),
                'herwig': np.ones(data_features.shape[0]),        
            }

            fig,ax,_ = plot_utils.HistRoutine(feed_dict,
                                              xlabel=name[feature],
                                              binning=binning[feature],
                                              weights = weights,
                                              label_loc='upper left',
                                              plot_ratio=True,
                                              reference_name='herwig',
                                              )
            fig.savefig('{}/omnifold_iter_{}_feat_{}.pdf'.format(flags.plot_folder,flags.num_iter,feature))

            d,derr = get_triangle_distance(feed_dict,weights,binning[feature],
                                           alternative_name=unfolded_name+'_fine_tune')
            print("OmniLearn feat {}: {} +- {}".format(name[feature],d,derr))

            d,derr = get_triangle_distance(feed_dict,weights,binning[feature],
                                           alternative_name=unfolded_name+'_baseline')
            print("Baseline feat {}: {} +- {}".format(name[feature],d,derr))
