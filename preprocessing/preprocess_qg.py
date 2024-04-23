import pandas as pd
import h5py
import os
import numpy as np
from optparse import OptionParser
import energyflow as ef

#Preprocessing for the top tagging dataset
def preprocess(data,folder,nparts=100):

    x,y = data
    particles = x[:,:,:-1]
    pid = x[:,:,-1]
    p4 = ef.p4s_from_ptyphipids(x)
        
    jets = np.sum(p4,axis=1)
    
    jet_e = jets[:,0]
    eta = ef.etas_from_p4s(jets)
    jets = ef.ptyphims_from_p4s(jets)
    jets[:,1]=eta
    #jets[:,3]=jet_e


    mask = particles[:,:,0]>0    

    p_e = particles[:,:,0]*np.cosh(particles[:,:,1])*mask
    particles = np.concatenate([particles,p_e[:,:,None]],-1)[:,:nparts]
    pid = pid[:,:nparts]


    NFEAT=13
    points = np.zeros((particles.shape[0],nparts,NFEAT))


    delta_phi = particles[:,:,2] - jets[:,None,2]
    delta_phi[delta_phi>np.pi] -=  2*np.pi
    delta_phi[delta_phi<= - np.pi] +=  2*np.pi


    points[:,:,0] = (particles[:,:,1] - jets[:,None,1])
    points[:,:,1] = delta_phi
    points[:,:,2] = np.ma.log(1.0 - particles[:,:,0]/jets[:,None,0]).filled(0)    
    points[:,:,3] = np.ma.log(particles[:,:,0]).filled(0)
    points[:,:,4] = np.ma.log(1.0 - particles[:,:,3]/jet_e[:,None]).filled(0)
    points[:,:,5] = np.ma.log(particles[:,:,3]).filled(0)
    points[:,:,6] = np.hypot(points[:,:,0],points[:,:,1])
    points[:,:,7] = np.sign(pid)* (pid!=22) * (pid!=130)
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
    with h5py.File('{}/train_qg.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[:1600000])
        dset = fh5.create_dataset('jet', data= jets[:1600000])
        dset = fh5.create_dataset('pid', data=y[:1600000])

    with h5py.File('{}/val_qg.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[1600000:1800000])
        dset = fh5.create_dataset('jet', data=jets[1600000:1800000])
        dset = fh5.create_dataset('pid', data=y[1600000:1800000])

    with h5py.File('{}/test_qg.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[1800000:])
        dset = fh5.create_dataset('jet', data=jets[1800000:])
        dset = fh5.create_dataset('pid', data=y[1800000:])


if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=100, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/PET/QG', help="Folder containing input files")

    (flags, args) = parser.parse_args()
        

    samples_path = flags.folder
    NPARTS = flags.npoints

    dataset_pythia = ef.qg_jets.load(num_data=-1, pad=True, ncol=4, generator='pythia',
                                     with_bc=False, cache_dir=samples_path)

    preprocess(dataset_pythia,samples_path)

