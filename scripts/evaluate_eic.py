import os
import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd
import argparse
import pickle
from PET_eicpythia import PET_eicpythia
import utils
import plot_utils
import matplotlib.pyplot as plt
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process jet data.")
    parser.add_argument("--dataset", type=str, default="eic", help="Dataset to use")
    parser.add_argument("--folder", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_argument("--mode", default="generator", help="Loss type to train the model: [all/classifier/generator]")
    parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
    parser.add_argument("--local", action='store_true', help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_argument("--simple", action='store_true', help="Use simplified head model")
    parser.add_argument("--talking_head", action='store_true', help="Use talking head attention")
    parser.add_argument("--layer_scale", action='store_true', help="Use layer scale in the residual connections")
    parser.add_argument("--sample", action='store_true', help="Sample from trained model")
    parser.add_argument("--plot_folder", default="../plots", help="Folder to save the outputs")
    return parser.parse_args()

def get_data_info(flags):
    test = utils.EicPythiaDataLoader(os.path.join(flags.folder,'EIC_Pythia','val_eic.h5'), rank=hvd.rank(), size=hvd.size())            
    return test


def load_data_and_model(flags):
    
    test = get_data_info(flags)
    model = PET_eicpythia(num_feat=test.num_feat,
                          num_jet=test.num_jet,
                          num_classes=test.num_classes,
                          num_part=test.num_part,
                          local=flags.local,
                          num_layers=flags.num_layers,
                          drop_probability=flags.drop_probability,
                          simple=flags.simple, layer_scale=flags.layer_scale,
                          talking_head=flags.talking_head,
                          mode=flags.mode, fine_tune=False, model_name=None)
    
    model_name = os.path.join(flags.folder, 'checkpoints', utils.get_model_name(flags, flags.fine_tune))
    model.load_weights(model_name)
    return test, model


def sample_data(test, model, flags, sample_name):
    """ Sample data using the model and save to file. """
    y, j = test.y[:], None
    
    nsplit = 10
    p, j = model.generate(y, jets=j, nsplit=nsplit,use_tqdm=hvd.rank()==0)
    p = test.revert_preprocess(p, p[:, :, 2] != 0)
    j = test.revert_preprocess_jet(j)

    particles_gen = hvd.allgather(tf.constant(p)).numpy()
    jets_gen = hvd.allgather(tf.constant(j)).numpy()
    y = hvd.allgather(tf.constant(y)).numpy()

    if hvd.rank() == 0:
        with h5.File(sample_name, "w") as h5f:
            h5f.create_dataset("data", data=particles_gen)
            h5f.create_dataset("jet", data=jets_gen)
            h5f.create_dataset("pid", data=y)
            
def get_generated_data(sample_name):
    with h5.File(sample_name,"r") as h5f:
        jets_gen = h5f['jet'][:]
        particles_gen = h5f['data'][:]
        
    def undo_pt(x):
        x[:,:,2] = 1.0 - np.exp(particles_gen[:,:,2])
        return x

    mask_gen = particles_gen[:,:,2]!=0
    #undo log transform for pt
    particles_gen = undo_pt(particles_gen)
    particles_gen = particles_gen*mask_gen[:,:,None]

    return jets_gen, particles_gen


def get_from_dataloader(test,nevts=-1):
    #Load eval samples for metric calculation
    X,flavour = test.data_from_file(test.files[0],preprocess=True)
    particles,jets,mask = X[0], X[3], X[2]

    
    
    particles = test.revert_preprocess(particles,mask)
    jets = test.revert_preprocess_jet(jets)
    particles[:,:,2] = 1.0 - np.exp(particles[:,:,2])
    #only keep the first 3 features
    if nevts<0:
        nevts = jets.shape[0]
        
    particles = particles[:nevts]*mask[:nevts,:,None]
    jets = jets[:nevts]
    return jets, particles


def plot(jet1,jet2,var_names,title,plot_folder):
    for ivar in range(len(var_names)):                
        feed_dict = {
            'eic_truth':jet1[:,ivar],
            'eic_gen':  jet2[:,ivar]
        }
            
        
        fig,gs,binning = plot_utils.HistRoutine(feed_dict,xlabel=var_names[ivar],
                                                plot_ratio=True,
                                                reference_name='eic_truth',
                                                ylabel= 'Normalized entries')

        ax0 = plt.subplot(gs[0])     
        fig.savefig('{}/EIC_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')



def plot_results(jets, jets_gen, particles, particles_gen, flags):
    """ Plot the results using the utility functions. """

    plot(jets, jets_gen, title='Jet', var_names=['electron $p_T$ [GeV]','electron $\eta$',
                                                 'Multiplicity'],
         plot_folder=flags.plot_folder)

    #Mask zero-padded particles
    particles_gen=particles_gen.reshape((-1,particles_gen.shape[-1]))
    mask_gen = particles_gen[:,2]!=0.
    particles_gen=particles_gen[mask_gen]
    particles=particles.reshape((-1,particles.shape[-1]))
    mask = particles[:,2]!=0.
    particles=particles[mask]

    plot(particles, particles_gen, title='Particle', var_names=['$\eta_{rel}$', '$\phi_{rel}$',
                                                                '$p_{Trel}$ [GeV]','is electron',
                                                                'is pion','is kaon'],
         plot_folder=flags.plot_folder)

def main():
    plot_utils.SetStyle()
    utils.setup_gpus()
    if hvd.rank()==0:logging.info("Horovod and GPUs initialized successfully.")
    flags = parse_arguments()
    sample_name = os.path.join(flags.folder, 'EIC_Pythia',
                               utils.get_model_name(flags, flags.fine_tune).replace(".weights.h5", ".h5"))
    
    if flags.sample:
        if hvd.rank()==0:logging.info("Sampling the data.")
        test, model = load_data_and_model(flags)
        sample_data(test, model, flags, sample_name)
    else:
        if hvd.rank()==0:logging.info("Loading saved samples.")
        # Load and process data, generate plots, etc.        
        test = get_data_info(flags)
        jets, particles = get_from_dataloader(test)
        jets_gen, particles_gen = get_generated_data(sample_name)
        print(particles_gen.shape,particles.shape)
        # Plot results
        plot_results(jets, jets_gen, particles, particles_gen, flags)

if __name__ == '__main__':
    main()


