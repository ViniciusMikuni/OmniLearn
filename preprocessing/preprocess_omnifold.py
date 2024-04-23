import pandas as pd
import h5py
import os
import numpy as np
from optparse import OptionParser
import energyflow as ef

def convert_pid(pid,mask):
    pid_dict = {
        0.0:22,
        0.1:211,
        0.2:-211,
        0.3:130,
        0.4:11,
        0.5:-11,
        0.6:13,        
        0.7:-13,
        0.8:321,
        0.9:-321,
        1.0:2212,
        1.1:-2212,
        1.2:2112,
        1.3:-2112,
        }
    for value in pid_dict:
        pid[pid==value] = pid_dict[value]

    return pid*mask


def get_substructure_obs(dataset):
    feature_names = ['widths','mults','sdms','zgs','tau2s']
    gen_features = [dataset['gen_jets'][:,3]]
    sim_features = [dataset['sim_jets'][:,3]]
    for feature in feature_names:
        gen_features.append(dataset['gen_'+feature])
        sim_features.append(dataset['sim_'+feature])


    gen_features = np.stack(gen_features,-1)
    sim_features = np.stack(sim_features,-1)
    #ln rho
    gen_features[:,3] = 2*np.ma.log(np.ma.divide(gen_features[:,3],dataset['gen_jets'][:,0]+10**-100).filled(0)).filled(0)
    sim_features[:,3] = 2*np.ma.log(np.ma.divide(sim_features[:,3],dataset['sim_jets'][:,0]+10**-100).filled(0)).filled(0)
    #tau21
    gen_features[:,5] = gen_features[:,5]/(10**-50 + gen_features[:,1])
    sim_features[:,5] = sim_features[:,5]/(10**-50 + sim_features[:,1])

    return sim_features, gen_features

#Preprocessing for the top tagging dataset
def preprocess(p,jets,folder,nparts=100):
    particles = p[:,:,:-1]
    mask = particles[:,:,0]>0
    particles[:,:,0] = 100.*particles[:,:,0]
    particles[:,:,1] = particles[:,:,1] + jets[:,None,1]
    particles[:,:,2] = particles[:,:,2] + jets[:,None,2]
    pid = p[:,:,-1]
    pid = convert_pid(pid,mask)
    p4 = ef.p4s_from_ptyphipids(np.concatenate([particles,pid[:,:,None]],-1))

    jets = np.sum(p4,axis=1)
    
    jets_e = jets[:,0]
    eta = ef.etas_from_p4s(jets)
    jets = ef.ptyphims_from_p4s(jets)
    jets[:,1]=eta

    
    p_e = particles[:,:,0]*np.cosh(particles[:,:,1])*mask
    particles = np.concatenate([particles,p_e[:,:,None]],-1)[:,:nparts]
    pid = pid[:,:nparts]
        
    NFEAT=13
    points = np.zeros((particles.shape[0],nparts,NFEAT))

    delta_phi = particles[:,:,2] - jets[:,None,2]
    delta_phi[delta_phi>np.pi] -=  2*np.pi
    delta_phi[delta_phi<= - np.pi] +=  2*np.pi
    


    points[:,:,0] = particles[:,:,1] - jets[:,None,1]
    points[:,:,1] = delta_phi
    points[:,:,2] = np.ma.log(1.0 - particles[:,:,0]/jets[:,None,0]).filled(0)        
    points[:,:,3] = np.ma.log(particles[:,:,0]).filled(0)
    points[:,:,4] = np.ma.log(1.0 - particles[:,:,3]/jets_e[:,None]).filled(0)
    points[:,:,5] = np.ma.log(particles[:,:,3]).filled(0)
    points[:,:,6] = np.hypot(points[:,:,0],points[:,:,1])
    points[:,:,7] = (np.sign(pid)) * (pid!=22) * (pid!=130)
    points[:,:,8] = (np.abs(pid) == 211) | (np.abs(pid) == 321) | (np.abs(pid) == 2212)
    points[:,:,9] = (np.abs(pid)==130) | (np.abs(pid) == 2112) | (pid == 0)
    points[:,:,10] = np.abs(pid)==22
    points[:,:,11] = np.abs(pid)==11
    points[:,:,12] = np.abs(pid)==13

    mult = np.sum(mask,-1)
    points*=mask[:,:nparts,None]

    #delete phi
    jets = np.delete(jets,2,axis=1)
    jets = np.concatenate([jets,mult[:,None]],-1)

    return points, jets

if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=101, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/PET/OmniFold', help="Folder containing input files")
    parser.add_option("--sample", type="string", default='pythia', help="sample to use")

    (flags, args) = parser.parse_args()
        

    samples_path = flags.folder
    NPARTS = flags.npoints


    if flags.sample == 'pythia':
        dataset_pythia = ef.zjets_delphes.load('Pythia21', num_data=-1, pad=True, cache_dir=samples_path,
                                               source='zenodo', which='all',
                                               include_keys=None, exclude_keys=None)
        
        reco_subs, gen_subs = get_substructure_obs(dataset_pythia)
        reco_parts, reco_jets = preprocess(dataset_pythia['sim_particles'],
                                           dataset_pythia['sim_jets'],
                                           samples_path,flags.npoints)
        gen_parts, gen_jets = preprocess(dataset_pythia['gen_particles'],
                                         dataset_pythia['gen_jets'],
                                         samples_path,flags.npoints)

        with h5py.File('{}/train_pythia.h5'.format(flags.folder), "w") as fh5:
            dset = fh5.create_dataset('reco', data=reco_parts[:1600000])
            dset = fh5.create_dataset('gen', data=gen_parts[:1600000])
            dset = fh5.create_dataset('reco_jets', data=reco_jets[:1600000])
            dset = fh5.create_dataset('gen_jets', data=gen_jets[:1600000])
            dset = fh5.create_dataset('reco_subs', data=reco_subs[:1600000])
            dset = fh5.create_dataset('gen_subs', data=gen_subs[:1600000])

        with h5py.File('{}/test_pythia.h5'.format(flags.folder), "w") as fh5:
            dset = fh5.create_dataset('reco', data=reco_parts[1200000:])
            dset = fh5.create_dataset('gen', data=gen_parts[1200000:])
            dset = fh5.create_dataset('reco_jets', data=reco_jets[1200000:])
            dset = fh5.create_dataset('gen_jets', data=gen_jets[1200000:])
            dset = fh5.create_dataset('reco_subs', data=reco_subs[1200000:])
            dset = fh5.create_dataset('gen_subs', data=gen_subs[1200000:])


    else:
    
        dataset_herwig = ef.zjets_delphes.load('Herwig', num_data=-1, pad=True, cache_dir=samples_path,
                                               source='zenodo', which='all',
                                               include_keys=None, exclude_keys=None)
        reco_subs, gen_subs = get_substructure_obs(dataset_herwig)
        reco_parts,reco_jets = preprocess(dataset_herwig['sim_particles'],
                                          dataset_herwig['sim_jets'],
                                          samples_path,flags.npoints)
        gen_parts, gen_jets = preprocess(dataset_herwig['gen_particles'],
                                         dataset_herwig['gen_jets'],
                                         samples_path,flags.npoints)
        
        with h5py.File('{}/train_herwig.h5'.format(flags.folder), "w") as fh5:
            dset = fh5.create_dataset('reco', data=reco_parts[:1600000])
            dset = fh5.create_dataset('gen', data=gen_parts[:1600000])
            dset = fh5.create_dataset('reco_jets', data=reco_jets[:1600000])
            dset = fh5.create_dataset('gen_jets', data=gen_jets[:1600000])
            dset = fh5.create_dataset('reco_subs', data=reco_subs[:1600000])
            dset = fh5.create_dataset('gen_subs', data=gen_subs[:1600000])


        with h5py.File('{}/test_herwig.h5'.format(flags.folder), "w") as fh5:
            dset = fh5.create_dataset('reco', data=reco_parts[1200000:])
            dset = fh5.create_dataset('gen', data=gen_parts[1200000:])
            dset = fh5.create_dataset('reco_jets', data=reco_jets[1200000:])
            dset = fh5.create_dataset('gen_jets', data=gen_jets[1200000:])    
            dset = fh5.create_dataset('reco_subs', data=reco_subs[1200000:])
            dset = fh5.create_dataset('gen_subs', data=gen_subs[1200000:])

