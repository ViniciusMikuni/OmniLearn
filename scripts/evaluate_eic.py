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
#keeping track of the number of variables plotted
base_vars = 4
add_vars = 2

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process elec data.")
    parser.add_argument("--dataset", type=str, default="eic", help="Dataset to use")
    parser.add_argument("--folder", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_argument("--mode", default="generator", help="Loss type to train the model: [all/classifier/generator]")
    parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
    parser.add_argument("--local", action='store_true', help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--nevts", type=int, default=-1, help="Number of events to load")
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
    
    nsplit = 20
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
            
def get_generated_data(sample_name,nevts=-1):

    with h5.File(sample_name,"r") as h5f:
        jets_gen = h5f['jet'][:]
        particles_gen = h5f['data'][:]
    if nevts>0:
        jets_gen = jets_gen[:nevts]
        particles_gen = particles_gen[:nevts]
        
    def undo_pt(x):
        x[:,:,2] = np.exp(particles_gen[:,:,2])
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
    particles[:,:,2] = np.exp(particles[:,:,2])
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
                                                logy=ivar==0,
                                                reference_name='eic_truth',
                                                ylabel= 'Normalized entries')

        ax0 = plt.subplot(gs[0])     

        p_dict = {"Particle_0" : "p",
                  "Particle_1" : "n",
                  "Particle_2" : "$K^+$",
                  "Particle_3" : '$\pi^{+}$',
                  "Particle_4" : 'neutrino',
                  "Particle_5" : '$\mu$',
                  "Particle_6" : 'e$^{-}$',
                  "Particle_7" : '$\gamma$',
                  "Particle_8" : '$\pi^{0}$',
                  }
        if title in p_dict:
            plt.title(p_dict[title], fontsize=35)

        fig.savefig('{}/EIC_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')
        plt.close()

def get_abs_eta(particles, electron):
    eta = particles[:,:,0] - electron[:,1,None]  #abs. eta
    return np.concatenate([eta[:,:,None], particles],-1)  #put abs eta as last index
    return particles

def get_z(particles,electron):
    z = pT_to_z(particles[:,:,3]*electron[:,0,None], particles[:,:,0])  #abs pT. abs eta done above
    z_ele = pT_to_z(electron[:,0,None], electron[:,1,None])
    sum_z = z_ele + np.sum(z,1,keepdims=True)
    return np.concatenate([np.ma.log10(z[:,:,None]).filled(0),particles],-1), np.concatenate([np.log10(z_ele),sum_z,electron],-1)

def get_mass(particles):
    mask = particles[:,:,2]!=0
    mass_vals = np.array([
         0.938,
         0.939,
         0.493,
         0.139,
         0.0,
         0.105,
         0.511e-5,
         0.0,
         0.497,
    ])
    mass = mass_vals[np.argmax(particles[:,:,5:],-1)]
    return mass*mask

def get_z_mass(particles,electron):
    mass = get_mass(particles)
    z = pT_to_z(particles[:,:,3]*electron[:,0,None], particles[:,:,0],mass)  #abs pT. abs eta done above
    z_ele = pT_to_z(electron[:,0,None], electron[:,1,None],mass = 0.511e-3*np.ones((electron.shape[0],1)))
    sum_z = z_ele + np.sum(z,1,keepdims=True)
    print(sum_z)
    return np.concatenate([np.ma.log10(z[:,:,None]).filled(0),particles],-1), np.concatenate([np.log10(z_ele),sum_z,electron],-1)


def pT_to_z(pT,eta,mass=None):
    eProton = 275
    eElectron = 10
    sqrt_s = np.sqrt(4*eProton*eElectron)
    if mass is None:
        return 2.*pT*np.cosh(eta)/sqrt_s
    else:
        m_T = np.sqrt(pT**2 + mass**2)
        y = 0.5 * np.ma.log((np.cosh(eta) + (pT/m_T) * np.sinh(eta)) / (np.cosh(eta) - (pT/m_T) * np.sinh(eta))).filled(0)
        return 2.*m_T*np.cosh(y)/sqrt_s

def plot_results(jets, jets_gen, particles, particles_gen, flags):
    """ Plot the results using the utility functions. """

    particles = get_abs_eta(particles, jets) #rel to abs eta, concats scattered e-
    particles_gen = get_abs_eta(particles_gen, jets_gen)

    particles, jets = get_z(particles,jets)
    #particles, jets = get_z_mass(particles,jets)
    particles_gen, jets_gen = get_z(particles_gen,jets_gen)


    ele_var_names = ['scattered e$^{-}$ $\log_{10}$(z)',
                     '$\sum_{i\in event} z_i$',
                     'scattered e$^{-}$ $p_T$ [GeV]',
                     'scattered e$^{-}$ $\eta$','Multiplicity']

    plot(jets, jets_gen, title='Electron',
         var_names=ele_var_names, plot_folder=flags.plot_folder)
    
    #Mask zero-padded particles
    particles_gen=particles_gen.reshape((-1,particles_gen.shape[-1]))
    particles_gen=particles_gen[particles_gen[:,4]!=0.]
    particles=particles.reshape((-1,particles.shape[-1]))
    particles=particles[particles[:,4]!=0.]
    
    #Inclusive plots with all particles
    part_var_names = ['all $\log_{10}$(z)','all $\eta_{abs}$',
                      'all $\eta_{rel}$', 'all $\phi_{rel}$',
                      'all $p_{Trel}$ [GeV]',
                      'charge','is proton','is neutron','is kaon',
                      'is pion', 'is neutrino',
                      'is muon','is electron',
                      'is photon', 'is pi0']
    
    plot(particles, particles_gen,title=f'Particle',
         var_names=part_var_names,plot_folder=flags.plot_folder)
    
    #Separate plots for each type of particle
    particle_names = ['p','n','K$^{+}$',
                      '$\pi^{+}$', 'neutrino',
                      '$\mu$', 'e$^{-}$',
                      '$\gamma$', '$\pi^{0}$']



    for pid in range(len(particle_names)):
        mask_pid = particles[:,base_vars+add_vars+pid]==1
        print(f"Fraction of {particle_names[pid]} in dataset: {1.0*np.sum(mask_pid)/mask_pid.shape[0]}")
        mask_pid_gen = particles_gen[:,base_vars+add_vars+pid]==1
        #Mask zero-padded particles
        particles_gen_pid=mask_pid_gen[:,None]*particles_gen
        particles_gen_pid=particles_gen_pid[particles_gen_pid[:,4]!=0.]
        particles_pid=mask_pid[:,None]*particles
        particles_pid=particles_pid[particles_pid[:,4]!=0.]

        var_names = [f'{particle_names[pid]}' + ' $\log_{10}$(z)',
                     f'{particle_names[pid]}' + ' $\eta_{abs}$',
                     f'{particle_names[pid]}' + ' $\eta_{rel}$',
                     f'{particle_names[pid]}' + ' $\phi_{rel}$',
                     f'{particle_names[pid]}' + ' $p_{Trel}$ [GeV]']


        plot(particles_pid[:,:len(var_names)],
             particles_gen_pid[:,:len(var_names)],
             title=f'Particle_{pid}', var_names=var_names,
             plot_folder=flags.plot_folder)

    # Electron PLots, scattered + produced
    mask_electron = particles[:,base_vars+add_vars+6]==1
    all_ele = np.zeros((particles[:,0][mask_electron].shape[0]+len(jets),2))
    all_ele[:,0] = np.concatenate((particles[:,0][mask_electron], jets[:,0]))
    all_ele[:,1] = np.concatenate((particles[:,1][mask_electron], jets[:,-2]))

    mask_electron_gen = particles_gen[:,base_vars+add_vars+6]==1
    all_ele_gen = np.zeros((particles_gen[:,0][mask_electron_gen].shape[0]+len(jets_gen),2))
    all_ele_gen[:,0] = np.concatenate((particles_gen[:,0][mask_electron_gen], jets_gen[:,0]))
    all_ele_gen[:,1] = np.concatenate((particles_gen[:,1][mask_electron_gen], jets_gen[:,-2]))

    ele_vars = ["$\log_{10}$(z)",'$\eta_{abs}$']

    plot(all_ele, all_ele_gen,title=f'All_Electrons',
         var_names=ele_vars, plot_folder=flags.plot_folder)

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
        jets, particles = get_from_dataloader(test,flags.nevts)
        jets_gen, particles_gen = get_generated_data(sample_name,flags.nevts)
        print(particles_gen.shape,particles.shape)
        # Plot results
        plot_results(jets, jets_gen, particles, particles_gen, flags)

if __name__ == '__main__':
    main()


