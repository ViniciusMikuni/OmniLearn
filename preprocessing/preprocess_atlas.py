import pandas as pd
import h5py
import os
import numpy as np
from optparse import OptionParser
import energyflow as ef

#Preprocessing for the top tagging dataset
def clustering(data,folder,sample='train.h5',nevents=1000,nparts=100):

    npid = data['labels'][:nevents]
    weights = data['weights'][:nevents]
    particles = np.stack([
        data['fjet_clus_pt'][:nevents,:nparts]/1000.,
        data['fjet_clus_eta'][:nevents,:nparts],
        data['fjet_clus_phi'][:nevents,:nparts],
        data['fjet_clus_E'][:nevents,:nparts]/1000.,
    ],-1)

    jets = np.stack([
        data['fjet_pt'][:nevents]/1000.,
        data['fjet_eta'][:nevents],
        data['fjet_phi'][:nevents],
        data['fjet_m'][:nevents]/1000.,
    ],-1)

    jet_e = np.sum(data['fjet_clus_E'][:nevents]/1000,1)

    
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
    jets = np.delete(jets,2,axis=1)

    if 'train' in sample:
        with h5py.File('{}/train_atlas.h5'.format(folder), "w") as fh5:
            dset = fh5.create_dataset('data', data=points[:int(0.8*npid.shape[0])])
            dset = fh5.create_dataset('jet', data=jets[:int(0.8*npid.shape[0])])
            dset = fh5.create_dataset('pid', data=npid[:int(0.8*npid.shape[0])])
            dset = fh5.create_dataset('weights', data=weights[:int(0.8*npid.shape[0])])

        with h5py.File('{}/val_atlas.h5'.format(folder), "w") as fh5:
            dset = fh5.create_dataset('data', data=points[int(0.8*npid.shape[0]):])
            dset = fh5.create_dataset('jet', data=jets[int(0.8*npid.shape[0]):])
            dset = fh5.create_dataset('pid', data=npid[int(0.8*npid.shape[0]):])
            dset = fh5.create_dataset('weights', data=weights[int(0.8*npid.shape[0]):])
        
    else:            
        with h5py.File('{}/{}_atlas.h5'.format(folder,sample), "w") as fh5:
            dset = fh5.create_dataset('data', data=points)
            dset = fh5.create_dataset('jet', data=jets)
            dset = fh5.create_dataset('pid', data=npid)
            dset = fh5.create_dataset('weights', data=weights)


if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=120, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/PET/ATLASTOP', help="Folder containing input files")
    parser.add_option("--sample", type="string", default='test.h5', help="Input file name")

    (flags, args) = parser.parse_args()
        

    samples_path = flags.folder
    sample = flags.sample
    NPARTS = flags.npoints

    data = h5py.File(os.path.join(samples_path,sample),'r')
    
    clustering(data,samples_path,flags.sample.replace('.h5',''),data['labels'].shape[0],NPARTS)
