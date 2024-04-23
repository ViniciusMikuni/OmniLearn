import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import tensorflow as tf
from tensorflow import keras
import sys
import horovod.tensorflow.keras as hvd
import pickle
from PET_lhco import PET_lhco, Classifier
import utils
import plot_utils

import matplotlib.pyplot as plt
plot_utils.SetStyle()

def get_features(p,j):
    #Based on jet and particle features, determine the full set of kinematic inputs used in the standard classifier training
    mask = p[:,:,:,2]!=0
    p_pt = j[:,:,None,0]*(1.0 - np.exp(p[:,:,:,2]))        
    p_e = p_pt*np.cosh(p[:,:,:,0] + j[:,:,1,None])
    j_e = np.sum(p_e,2)[:,:,None]
    
    new_p = np.zeros((p.shape[0],p.shape[1],p.shape[2],7))
    new_p[:,:,:,:3] += p[:,:,:,:3]
    new_p[:,:,:,3] = np.ma.log(p_pt).filled(0)
    
    new_p[:,:,:,4] = np.ma.log(1.0 - p_e/j_e).filled(0)    
    new_p[:,:,:,5] = np.ma.log(p_e).filled(0)
    new_p[:,:,:,6] = np.hypot(new_p[:,:,:,0],new_p[:,:,:,1])
    
    return new_p*mask[:,:,:,None]


def plot(true,gen,nplots,title,plot_folder,names,weights=None):

    for ivar in range(nplots):                    
        feed_dict = {'true':true[:,ivar]}
        for sample in gen:
            feed_dict[sample] = gen[sample][:,ivar]

        if weights is not None:
            weight_dict = {'true':np.ones(true.shape[0])}
            for sample in weights:
                weight_dict[sample] = weights[sample]
        else:
            weight_dict = None
            

        fig,gs,_ = plot_utils.HistRoutine(feed_dict,xlabel=names[ivar],
                                          weights=weight_dict,
                                          plot_ratio=True,
                                          reference_name='true',
                                          ylabel= 'Normalized entries')
        
        ax0 = plt.subplot(gs[0])     
        # ax0.set_ylim(top=config.max_y)

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig('{}/lhco_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')

def sample_data(flags,folder,sample_name,use_SR=True,nevt=200000):
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','train_background_SB.h5'),rank=hvd.rank(),size=hvd.size(),nevts=nevt)

    if use_SR:
        #This assumes you have a file containing the list of mjj values you want to be generated
        y = test.LoadMjjFile(os.path.join(flags.folder,'LHCO'),'mjj_sample.h5',use_SR=True)[hvd.rank():nevt:hvd.size()]
        del train
        jets = None
    else:
        y = test.y
        jets = None
        

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

    nsplit = 25
    #25
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
        nevts = y.shape[0]
        with h5.File(os.path.join(folder,"train_"+sample_name),"w") as h5f:
            dset = h5f.create_dataset("data", data=particles_gen[:int(0.8*nevts)])
            dset = h5f.create_dataset("jet", data=jets_gen[:int(0.8*nevts)])
            dset = h5f.create_dataset("pid", data=y[:int(0.8*nevts)])

        with h5.File(os.path.join(folder,"test_"+sample_name),"w") as h5f:
            dset = h5f.create_dataset("data", data=particles_gen[int(0.8*nevts):])
            dset = h5f.create_dataset("jet", data=jets_gen[int(0.8*nevts):])
            dset = h5f.create_dataset("pid", data=y[int(0.8*nevts):])

def get_generated_particles(sample_name):
    with h5.File(sample_name,"r") as h5f:
        jets_gen = h5f['jet'][:]
        particles_gen = h5f['data'][:]
        mass_gen = h5f['pid'][:jets_gen.shape[0]]


    mask_gen = particles_gen[:,:,:,2]!=0
    particles_gen = particles_gen*mask_gen[:,:,:,None]

    return jets_gen, particles_gen, mass_gen

def get_from_file(test,nevts=-1):
    #Load eval samples for metric calculation
    X = test.X
    mjj = test.y
    jets = test.jet
    return jets, X, mjj
    


if __name__=='__main__':


    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--dataset", type="string", default="lhco", help="Folder containing input files")
    parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_option("--mode", type="string", default="generator", help="Loss type to train the model: available options are [all/classifier/generator]")
    
    parser.add_option('--fine_tune', action='store_true', default=False,help='Fine tune a model')
    parser.add_option('--SR', action='store_true', default=False,help='Generate SR data')
    #Model parameters
    parser.add_option('--local', action='store_true', default=False,help='Use local embedding')
    parser.add_option("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_option("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_option("--nevt", type=int, default=2000000, help="Number of events to generate")

    parser.add_option('--simple', action='store_true', default=False,help='Use simplified head model')
    parser.add_option('--talking_head', action='store_true', default=False,help='Use talking head attention instead of standard attention')
    parser.add_option('--layer_scale', action='store_true', default=False,help='Use layer scale in the residual connections')
    parser.add_option('--skip_classifier', action='store_true', default=False,help='Skip metric calculation')
    
    parser.add_option('--sample', action='store_true', default=False,help='Sample from trained model')
    parser.add_option('--weighted', action='store_true', default=False,help='Load weights to correct model prediction')
    parser.add_option("--plot_folder", type="string", default="../plots", help="Folder to save the outputs")
    
    (flags, args) = parser.parse_args()

    if flags.sample:
        sample_data(flags,folder = os.path.join(flags.folder,'LHCO'),
                    sample_name=utils.get_model_name(flags,flags.fine_tune).replace(".weights.h5","_{}.h5".format("SR" if flags.SR else "SB")),
                    use_SR=flags.SR,nevt = flags.nevt)
    else:
        gen_omni = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','train_'+utils.get_model_name(flags,True).replace(".weights.h5","_{}.h5".format("SR" if flags.SR else "SB"))))
        gen_pet = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','train_'+utils.get_model_name(flags,False).replace(".weights.h5","_{}.h5".format("SR" if flags.SR else "SB"))))
        
        jets_gen_omni, particles_gen_omni, mjj_gen_omni = get_from_file(gen_omni)
        jets_gen_pet, particles_gen_pet, mjj_gen_pet = get_from_file(gen_pet)
        
        test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','val_background_{}.h5'.format('SR' if flags.SR else 'SB')))        
        jets, particles, mjj = get_from_file(test)
            

        title = 'jet_{}'.format('SR' if flags.SR else 'SB')
        jets = jets.reshape(-1,test.num_jet)        
        jets_gen_omni = jets_gen_omni.reshape(-1,test.num_jet)
        jets_gen_pet = jets_gen_pet.reshape(-1,test.num_jet)
        
        
        
        jet_names = ['Jet p$_{T}$ [GeV]', 'Jet $\eta$', 'Jet $\phi$','Jet Mass [GeV]','Multiplicity']
        plot(jets,
             {'lhco_fine_tune':jets_gen_omni,
              'lhco':jets_gen_pet},
             title=title,names=jet_names,
             nplots=test.num_jet,plot_folder=flags.plot_folder)


        # particles_gen=particles_gen.reshape((-1,7))
        # mask_gen = particles_gen[:,2]!=0.
        # particles_gen=particles_gen[mask_gen]
        # particles=particles.reshape((-1,7))
        # mask = particles[:,2]!=0.
        # particles=particles[mask]
        # title = 'part_{}'.format('SR' if flags.SR else 'SB')
        # part_names = ['$\eta_{rel}$', '$phi_rel$', 'log($1 - p_{Trel}$)','log($p_{T}$)','log($1 - E_{rel}$)','log($E$)','$\Delta$R']
        # plot(particles,particles_gen,
        #      title=title,
        #      names = part_names,
        #      nplots=7,
        #      plot_folder=flags.plot_folder)
    
