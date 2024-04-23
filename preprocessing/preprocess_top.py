import pandas as pd
import h5py
import os
import numpy as np
from optparse import OptionParser
import energyflow as ef

#Preprocessing for the top tagging dataset
def clustering_sum(data,folder,sample='train',nevents=1000,nparts=100):

    npid = data[:nevents,-1]
    
    particles = data[:,0:200*4]
    particles=particles.reshape((data.shape[0],-1,4))

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
    
    with h5py.File('{}/{}_ttbar.h5'.format(folder,sample), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[:,:nparts])
        dset = fh5.create_dataset('jet', data=jets)
        dset = fh5.create_dataset('pid', data=npid)


if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=150, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/PET/TOP/', help="Folder containing input files")
    parser.add_option("--sample", type="string", default='train.h5', help="Input file name")

    (flags, args) = parser.parse_args()
        

    samples_path = flags.folder
    sample = flags.sample
    NPARTS = flags.npoints

    store = pd.HDFStore(os.path.join(samples_path,sample),'r')

    data = store['table'].values
    clustering_sum(data,samples_path,flags.sample.replace('.h5',''),data.shape[0],NPARTS)
