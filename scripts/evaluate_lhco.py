import os
import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd
import argparse
import pickle
from PET_lhco import PET_lhco, Classifier
import utils
import plot_utils
import matplotlib.pyplot as plt

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plot_utils.SetStyle()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process LHCO data.")
    parser.add_argument("--dataset", default="lhco", help="Folder containing input files")
    parser.add_argument("--folder", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_argument("--mode", default="generator", help="Loss type to train the model: [all/classifier/generator]")
    parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
    parser.add_argument("--SR", action='store_true', help="Generate SR data")
    parser.add_argument("--local", action='store_true', help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_argument("--nevt", type=int, default=2000000, help="Number of events to generate")
    parser.add_argument("--simple", action='store_true', help="Use simplified head model")
    parser.add_argument("--talking_head", action='store_true', help="Use talking head attention")
    parser.add_argument("--layer_scale", action='store_true', help="Use layer scale in the residual connections")
    parser.add_argument("--sample", action='store_true', help="Sample from trained model")
    parser.add_argument("--weighted", action='store_true', help="Load weights to correct model prediction")
    parser.add_argument("--plot_folder", default="../plots", help="Folder to save the outputs")
    return parser.parse_args()

def get_features(p, j):
    # Determine the full set of kinematic inputs based on jet and particle features
    mask = p[:, :, :, 2] != 0
    p_pt = j[:, :, None, 0] * (1.0 - np.exp(p[:, :, :, 2]))
    p_e = p_pt * np.cosh(p[:, :, :, 0] + j[:, :, 1, None])
    j_e = np.sum(p_e, 2)[:, :, None]

    new_p = np.zeros((p.shape[0], p.shape[1], p.shape[2], 7))
    new_p[:, :, :, :3] += p[:, :, :, :3]
    new_p[:, :, :, 3] = np.ma.log(p_pt).filled(0)
    new_p[:, :, :, 4] = np.ma.log(1.0 - p_e / j_e).filled(0)
    new_p[:, :, :, 5] = np.ma.log(p_e).filled(0)
    new_p[:, :, :, 6] = np.hypot(new_p[:, :, :, 0], new_p[:, :, :, 1])

    return new_p * mask[:, :, :, None]

def plot(data,title,plot_folder,names,weights=None):
    
    for ivar in range(len(names)):
        feed_dict = {}
        for sample in data:
            feed_dict[sample] = data[sample][:,ivar]

        if weights is not None:
            for sample in weights:
                weight_dict[sample] = weights[sample]
        else:
            weight_dict = None
            
        fig,gs,_ = plot_utils.HistRoutine(feed_dict,xlabel=names[ivar],
                                          weights=weight_dict,
                                          plot_ratio=True,
                                          reference_name='true',
                                          ylabel= 'Normalized entries')

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
            
        fig.savefig('{}/lhco_{}_{}.pdf'.format(plot_folder,title,ivar))


def load_data_sample(flags):
    test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','train_background_SB.h5'),
                                rank=hvd.rank(),size=hvd.size(),nevts=flags.nevt)

    if flags.SR:
        #This assumes you have a file containing the list of mjj values you want to be generated
        y = test.LoadMjjFile(os.path.join(flags.folder,'LHCO'),'mjj_sample.h5',use_SR=True)[hvd.rank():nevt:hvd.size()]
        del train
        jets = None
    else:
        y = test.y
        jets = None
    
    return test, jets, y


def load_model(flags,test):
    model = PET_lhco(num_feat=test.num_feat,
                 num_jet=test.num_jet,
                 num_classes=2,
                 num_part = test.num_part,
                 local = flags.local,
                 num_layers = flags.num_layers, 
                 drop_probability = flags.drop_probability,
                 simple = flags.simple, layer_scale = flags.layer_scale,
                 talking_head = flags.talking_head,
                 mode = flags.mode,
                 fine_tune = False,
                 model_name = None,
                 use_mean = flags.fine_tune,
                 )
    
    model_name = os.path.join(flags.folder,'checkpoints',utils.get_model_name(flags,flags.fine_tune))
    model.load_weights('{}'.format(model_name))
    return model


def sample_data(flags,folder,sample_name,nsplit=25):
    utils.setup_gpus()        

    test, jets, y = load_data_sample(flags)
    model = load_model(flags,test)

    p,j = model.generate(y,jets,nsplit=nsplit)
    j = test.revert_preprocess_jet(j)
    p = test.revert_preprocess(p,p[:,:,:,2]!=0)

    
    #Determine additional input features from the generated ones    
    p = get_features(p,j)
    particles_gen = hvd.allgather(tf.constant(p)).numpy()
    jets_gen = hvd.allgather(tf.constant(j)).numpy()
        
    y = test.revert_mjj(y)
    y = hvd.allgather(tf.constant(y)).numpy()
    
    if hvd.rank()==0:
        logger.info("Saving generated data.")
        nevts = y.shape[0]
        with h5.File(os.path.join(folder,"train_"+sample_name),"w") as h5f:
            dset = h5f.create_dataset("data", data=particles_gen[:int(0.8*nevts)])
            dset = h5f.create_dataset("jet", data=jets_gen[:int(0.8*nevts)])
            dset = h5f.create_dataset("pid", data=y[:int(0.8*nevts)])

        with h5.File(os.path.join(folder,"test_"+sample_name),"w") as h5f:
            dset = h5f.create_dataset("data", data=particles_gen[int(0.8*nevts):])
            dset = h5f.create_dataset("jet", data=jets_gen[int(0.8*nevts):])
            dset = h5f.create_dataset("pid", data=y[int(0.8*nevts):])


def main():
    flags = parse_arguments()
    if flags.sample:
        logger.info("Sampling the data.")
        sample_data(flags,folder = os.path.join(flags.folder,'LHCO'),
                    sample_name=utils.get_model_name(flags,flags.fine_tune).replace(
                        ".weights.h5","_{}.h5".format("SR" if flags.SR else "SB")))        
    else:
        logger.info("Loading saved samples.")
        # Load and process data, generate plots, etc.

        file_names = {
            'lhco_fine_tune': os.path.join(flags.folder,'LHCO','train_'+utils.get_model_name(flags,True).replace(".weights.h5","_{}.h5".format("SR" if flags.SR else "SB"))),
            'lhco': os.path.join(flags.folder,'LHCO','train_'+utils.get_model_name(flags,False).replace(".weights.h5","_{}.h5".format("SR" if flags.SR else "SB"))),
            'true': os.path.join(flags.folder,'LHCO','val_background_{}.h5'.format('SR' if flags.SR else 'SB')),
        }

        particles, jets = {},{}
        for file_name in file_names:
            test = utils.LHCODataLoader(file_names[file_name])
            jet, particle = test.jet, test.X
            #flatten features
            jets[file_name] = jet.reshape(-1,test.num_jet)
            particles[file_name] = particle.reshape((-1,particle.shape[-1]))
            #remove zero-padded entries
            mask = particles[file_name][:,2]!=0.
            particles[file_name]=particles[file_name][mask]
            
        logger.info("Plotting jets.")
        title = 'jet_{}'.format('SR' if flags.SR else 'SB')        
        jet_names = ['Jet p$_{T}$ [GeV]', 'Jet $\eta$', 'Jet $\phi$','Jet Mass [GeV]','Multiplicity']
        plot(jets,title=title,names=jet_names,plot_folder=flags.plot_folder)

        logger.info("Plotting particles.")
        title = 'part_{}'.format('SR' if flags.SR else 'SB')
        part_names = ['$\eta_{rel}$', '$\phi_{rel}$', 'log($1 - p_{Trel}$)','log($p_{T}$)','log($1 - E_{rel}$)','log($E$)','$\Delta$R']
        plot(particles,title=title,names = part_names,plot_folder=flags.plot_folder)
    
if __name__ == '__main__':
    main()

