import pandas as pd
import h5py
import os
import numpy as np
from optparse import OptionParser
import energyflow as ef
from sklearn.utils import shuffle

#Preprocessing for the top tagging dataset
def clustering_sum(data_top,data_qcd,folder,nparts=100,is_train = False):

    particles = np.concatenate([data_top,data_qcd])
    particles =  ef.p4s_from_ptyphims(particles)

    
    npid = np.concatenate([np.ones(data_top.shape[0],dtype=np.int32),
                           np.zeros(data_qcd.shape[0],dtype=np.int32)])
    

    jets = np.sum(particles,axis=1)
    jet_e = jets[:,0]
    eta = ef.etas_from_p4s(jets)
    jets = ef.ptyphims_from_p4s(jets)
    jets[:,1]=eta
    

    p_e = particles[:,:,0]
    particles = ef.ptyphims_from_p4s(particles)
    particles[:,:,3]=p_e
    mask = particles[:,:,0]>0

    jets = np.concatenate([jets,np.sum(mask,-1)[:,None]],-1)
    
    NFEAT=7
    points = np.zeros((particles.shape[0],particles.shape[1],NFEAT))


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
    

    points*=mask[:,:,None]
    points = points[:,:nparts]

    jets = np.delete(jets,2,axis=1)

    points,jets,npid = shuffle(points,jets,npid)

    if is_train:    
        with h5py.File('{}/train_ttbar.h5'.format(folder,sample), "w") as fh5:
            dset = fh5.create_dataset('data', data=points[:1_600_000,:nparts])
            dset = fh5.create_dataset('jet', data=jets[:1_600_000])
            dset = fh5.create_dataset('pid', data=npid[:1_600_000])


        with h5py.File('{}/val_ttbar.h5'.format(folder), "w") as fh5:
            dset = fh5.create_dataset('data', data=points[1_600_000:,:nparts])
            dset = fh5.create_dataset('jet', data=jets[1_600_000:])
            dset = fh5.create_dataset('pid', data=npid[1_600_000:])

    else:
        with h5py.File('{}/test_ttbar.h5'.format(folder), "w") as fh5:
            dset = fh5.create_dataset('data', data=points[:,:nparts])
            dset = fh5.create_dataset('jet', data=jets)
            dset = fh5.create_dataset('pid', data=npid)

        

if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=128, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/PET/Opt/top/raw', help="Folder containing input files")
    parser.add_option("--sample", type="string", default='train_nsamples1M_trunc_5000.h5', help="Input file name")

    (flags, args) = parser.parse_args()
        

    samples_path = flags.folder
    sample = flags.sample
    NPARTS = flags.npoints


    data_top = h5py.File(os.path.join(samples_path,sample))['raw'][:]
    data_qcd = h5py.File(os.path.join(samples_path.replace("top","qcd"),sample))['raw'][:]
    clustering_sum(data_top,data_qcd,samples_path,NPARTS,is_train = 'train' in sample)
