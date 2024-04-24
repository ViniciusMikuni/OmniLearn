import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import tensorflow as tf
from tensorflow import keras
import sys
import horovod.tensorflow.keras as hvd
import pickle
from PET_jetnet import PET_jetnet
import utils
import plot_utils

import matplotlib.pyplot as plt
plot_utils.SetStyle()

names = ['g','q','t','w','z']
def sample_data(flags,sample_name,keep_top=False,ideal=False):
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        
    if flags.dataset == 'jetnet150':
        test = utils.JetNetDataLoader(os.path.join(flags.folder,'JetNet','test_150.h5'),rank = hvd.rank(),size = hvd.size(),big=True)
    
    elif flags.dataset == 'jetnet30':
        test = utils.JetNetDataLoader(os.path.join(flags.folder,'JetNet','test_30.h5'),rank = hvd.rank(),size = hvd.size())


    model = PET_jetnet(num_feat=test.num_feat,
                 num_jet=test.num_jet,
                 num_classes=test.num_classes,
                 num_part = test.num_part,
                 local = flags.local,
                 num_layers = flags.num_layers, 
                 drop_probability = flags.drop_probability,
                 simple = flags.simple, layer_scale = flags.layer_scale,
                 talking_head = flags.talking_head,
                 mode = flags.mode,
                 fine_tune = False,
                 model_name = None,
                 use_mean=flags.fine_tune,
                 )
    
    model_name = os.path.join(flags.folder,'checkpoints',utils.get_model_name(flags,flags.fine_tune))
    model.load_weights('{}'.format(model_name))
    y = test.y[:]
    
    if ideal:
        j = test.preprocess_jet(test.jet)
    else:
        file_name = '/pscratch/sd/v/vmikuni/PET/JetNet/PET_jetnet150_8_local_layer_scale_token_baseline_generator.h5'
        y = h5.File(file_name)['pid'][:]
        j = None
        
    if keep_top:
        if j is not None: j = j[np.argmax(y,-1) == 2]
        y = y[np.argmax(y,-1) == 2]

    if '150' in sample_name:
        nsplit = 400
    else:
        nsplit = 10
    p,j = model.generate(y,jets=j,nsplit=nsplit)

    p = test.revert_preprocess(p,p[:,:,2]!=0)
    j = test.revert_preprocess_jet(j)
    
    # if hvd.rank()==0:
    #     with h5.File(sample_name,"w") as h5f:
    #         dset = h5f.create_dataset("data", data=p)
    #         dset = h5f.create_dataset("jet", data=j)
    #         dset = h5f.create_dataset("pid", data=y)
            
    particles_gen = hvd.allgather(tf.constant(p)).numpy()
    jets_gen = hvd.allgather(tf.constant(j)).numpy()
    y = hvd.allgather(tf.constant(y)).numpy()
    
    if hvd.rank()==0:
        with h5.File(sample_name,"w") as h5f:
            dset = h5f.create_dataset("data", data=particles_gen)
            dset = h5f.create_dataset("jet", data=jets_gen)
            dset = h5f.create_dataset("pid", data=y)

def get_generated_particles(sample_name):
    with h5.File(sample_name,"r") as h5f:
        jets_gen = h5f['jet'][:]
        particles_gen = h5f['data'][:,:,:3]        
        flavour_gen = h5f['pid'][:jets_gen.shape[0]]

    def undo_pt(x):
        x[:,:,2] = 1.0 - np.exp(particles_gen[:,:,2])
        x[:,:,2] = np.clip(x[:,:,2],0.0,1.0)
        return x

    mask_gen = particles_gen[:,:,2]!=0
    #undo log transform
    particles_gen = undo_pt(particles_gen)
    particles_gen = particles_gen*mask_gen[:,:,None]

    return jets_gen, particles_gen, flavour_gen

def get_from_file(test,keep_top=False,nevts=-1):
    #Load eval samples for metric calculation
    X,flavour = test.data_from_file(test.files[0],preprocess=True)
    particles,jets,mask = X[0], X[3], X[2]
    
    if keep_top:
        mask_pid = np.argmax(flavour,-1) == 2    
        particles=particles[mask_pid]
        jets=jets[mask_pid]
        flavour=flavour[mask_pid]
        mask = mask[mask_pid]
        
    particles = test.revert_preprocess(particles,mask)
    jets = test.revert_preprocess_jet(jets)
    particles[:,:,2] = 1.0 - np.exp(particles[:,:,2])
    #only keep the first 3 features
    if nevts<0:
        nevts = jets.shape[0]
        
    particles = particles[:nevts,:,:3]*mask[:nevts,:,None]
    jets = jets[:nevts]
    flavour = flavour[:nevts]

    return jets, particles, flavour
    


if __name__=='__main__':


    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--dataset", type="string", default="jetnet150", help="Folder containing input files")
    parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_option("--mode", type="string", default="generator", help="Loss type to train the model: available options are [all/classifier/generator]")
    
    parser.add_option('--fine_tune', action='store_true', default=False,help='Fine tune a model')
    
    #Model parameters
    parser.add_option('--local', action='store_true', default=False,help='Use local embedding')
    parser.add_option("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_option("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_option('--simple', action='store_true', default=False,help='Use simplified head model')
    parser.add_option('--talking_head', action='store_true', default=False,help='Use talking head attention instead of standard attention')
    parser.add_option('--layer_scale', action='store_true', default=False,help='Use layer scale in the residual connections')
    parser.add_option('--skip_metric', action='store_true', default=False,help='Skip metric calculation')
    
    parser.add_option('--sample', action='store_true', default=False,help='Sample from trained model')
    parser.add_option('--top', action='store_true', default=False,help='Sample only top quarks')
    parser.add_option('--ideal', action='store_true', default=False,help='Use tru jets for the generation')
    parser.add_option("--plot_folder", type="string", default="../plots", help="Folder to save the outputs")
    
    (flags, args) = parser.parse_args()

    sample_name = os.path.join(flags.folder,'JetNet',
                               utils.get_model_name(flags,flags.fine_tune,add_string='_ideal' if flags.ideal else "").replace(".weights.h5",".h5"))
    
    if flags.sample:
        sample_data(flags,sample_name,keep_top=flags.top,ideal=flags.ideal)
    else:
        print("Loading file: {}".format(sample_name))
        jets_gen, particles_gen, flavour_gen = get_generated_particles(sample_name)
        if flags.dataset == 'jetnet150':
            test = utils.JetNetDataLoader(os.path.join(flags.folder,'JetNet','test_150.h5'),big=True)   
        elif flags.dataset == 'jetnet30':
            test = utils.JetNetDataLoader(os.path.join(flags.folder,'JetNet','test_30.h5'))

        jets, particles, flavour = get_from_file(test,keep_top = flags.top)
                
        plot_utils.plot(jets,jets_gen,flavour,flavour_gen,title='jet',names=names,
                        nplots=4,plot_folder=flags.plot_folder,is_big=flags.dataset == 'jetnet150')
    
        if not flags.skip_metric:
            from jetnet.evaluation.gen_metrics import w1p,w1m,w1efp,cov_mmd,fpnd
            from scipy.stats import wasserstein_distance

            print("Calculating metrics")

            with open(sample_name.replace('.h5','.txt'),'w') as f:
                for unique in np.unique(np.argmax(flavour,-1)):
                    mask = np.argmax(flavour,-1) == unique
                    mask_gen = np.argmax(flavour_gen,-1) == unique
                    print(names[unique])
                    f.write(names[unique])
                    f.write("\n")

                    mean_mass,std_mass = w1m(particles[mask], particles_gen[mask_gen])
                    print("W1M",1e3*mean_mass,1e3*std_mass)
                    f.write("{:.2f} $\pm$ {:.2f} & ".format(1e3*mean_mass,1e3*std_mass))            
                    mean,std = w1p(particles[mask], particles_gen[mask_gen])
                    print("W1P: ",1e3*np.mean(mean),1e3*mean,np.mean(std))
                    f.write("{:.2f} $\pm$ {:.2f} & ".format(1e3*np.mean(mean),1e3*np.mean(std)))
                    mean_efp,std_efp = w1efp(particles[mask], particles_gen[mask_gen])
                    print("W1EFP",1e5*np.mean(mean_efp),1e5*np.mean(std_efp))
                    f.write("{:.2f} $\pm$ {:.2f} & ".format(1e5*np.mean(mean_efp),1e5*np.mean(std_efp)))
                    if flags.dataset == 'jetnet150' or 'w' in names[unique] or 'z' in names[unique]:
                        #FPND only defined for 30 particles and not calculated for W and Z
                        pass
                    else:
                        fpnd_score = fpnd(particles_gen[mask_gen], jet_type=names[unique])
                        print("FPND", fpnd_score)
                        f.write("{:.2f} & ".format(fpnd_score))
                    
                    cov,mmd = cov_mmd(particles[mask],particles_gen[mask_gen],num_eval_samples=1000)
                    print("COV,MMD",cov,mmd)
                    f.write("{:.2f} & {:.2f} \\\\".format(cov,mmd))
                    f.write("\n")
            
        flavour = np.tile(np.expand_dims(flavour,1),(1,particles_gen.shape[1],1)).reshape((-1,flavour.shape[-1]))
        particles_gen=particles_gen.reshape((-1,3))
        mask_gen = particles_gen[:,2]>0.
        particles_gen=particles_gen[mask_gen]
        particles=particles.reshape((-1,3))
        mask = particles[:,2]>0.
        particles=particles[mask]
    
        flavour_gen = flavour[mask_gen]
        flavour = flavour[mask]

        
        plot_utils.plot(particles,particles_gen,
                        flavour,flavour_gen,
                        title='part',names=names,
                        nplots=3,
                        plot_folder=flags.plot_folder,
                        is_big=flags.dataset == 'jetnet150')
        

