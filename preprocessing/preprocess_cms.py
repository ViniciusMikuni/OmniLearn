import pandas as pd
import h5py
import os, gc
import numpy as np
from optparse import OptionParser
import energyflow as ef
from tqdm import tqdm

def pad_and_combine(arrays,M):
    F = arrays[0].shape[1] -1 #skip vertex information
    
    # Initialize the list to hold padded arrays
    padded_arrays = np.zeros((len(arrays),M,F))
    
    # Pad each subarray if necessary and append to padded_arrays
    for iarr, array in tqdm(enumerate(arrays), total = len(arrays), desc="Processing arrays"):        
        P = array.shape[0]        
        # Check if padding is needed (if P < M)
        if P < M:
            padded_arrays[iarr,:P] += array[:,:-1]
        else:
            padded_arrays[iarr] += array[:M,:-1]
    
    return  padded_arrays


def balance_classes(x, y, z):
    """
    Balances the classes in the dataset by randomly discarding entries from the more populous class.

    Parameters:
        x (numpy.ndarray): Input array of shape (N, P, F) where N is the number of samples,
                           P is the dimension of each sample, and F is the number of features.
        y (numpy.ndarray): Label array of shape (N,) where each entry is either 0 or 1.

    Returns:
        tuple: A tuple containing the balanced input array and label array.
    """
    # Find indices for each class
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]
    
    # Determine the minority class and its size
    if len(indices_0) > len(indices_1):
        minority_size = len(indices_1)
        indices_0 = np.random.choice(indices_0, minority_size, replace=False)
    else:
        minority_size = len(indices_0)
        indices_1 = np.random.choice(indices_1, minority_size, replace=False)
    
    # Combine and shuffle indices
    balanced_indices = np.concatenate([indices_0, indices_1])
    np.random.shuffle(balanced_indices)
    
    # Extract samples and labels corresponding to these indices
    x_balanced = x[balanced_indices]
    y_balanced = y[balanced_indices]
    z_balanced = z[balanced_indices]
    return x_balanced, y_balanced, z_balanced


#Preprocessing for the top tagging dataset
def preprocess(data,folder,nparts=100, use_pid = True):
    print("Creating labels")
    y = data.jets_i[:,-1]
    y[(np.abs(y)==1)|(np.abs(y)==2)|(np.abs(y)==3)] = 1 #uds
    y[y==21] = 0
    jets = np.stack([data.jets_f[:,0],data.jets_f[:,4],data.jets_f[:,2],data.jets_f[:,3]],-1)
    jets = jets[np.abs(y)<2]
    particles = data.particles[np.abs(y)<2]
    y = y[np.abs(y) < 2] #Reject c and b jets

    #Min pt cut for particles
    particles = np.asarray([part[ef.mod.filter_particles(part, pt_cut = 1)] for part in particles])
    print("Start preparing the dataset")
    del data
    gc.collect()

    particles = pad_and_combine(particles,nparts)

    print("Balancing classes")
    #Balance the number of signal and background events
    particles, y,jets = balance_classes(particles, y,jets)

    print("Total sample size after balancing: {}".format(particles.shape[0]))
    pid = particles[:,:,-1]

       
    jet_e = np.sqrt(jets[:,0]**2*np.cosh(jets[:,1])**2 + jets[:,3]**2)
    mask = particles[:,:,0]>0    

    p_e = particles[:,:,0]*np.cosh(particles[:,:,1])*mask
    particles[:,:,3] = p_e


    if use_pid:
        NFEAT=13
    else:
        NFEAT=7
        
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
    if use_pid:
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

    train_nevt = int(0.7*jets.shape[0])
    val_nevt = train_nevt + int(0.2*jets.shape[0])
    
    with h5py.File('{}/train_qgcms_pid.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[:train_nevt])
        dset = fh5.create_dataset('jet', data= jets[:train_nevt])
        dset = fh5.create_dataset('pid', data=y[:train_nevt])

    with h5py.File('{}/val_qgcms_pid.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[train_nevt:val_nevt])
        dset = fh5.create_dataset('jet', data=jets[train_nevt:val_nevt])
        dset = fh5.create_dataset('pid', data=y[train_nevt:val_nevt])

    with h5py.File('{}/test_qgcms_pid.h5'.format(folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=points[val_nevt:])
        dset = fh5.create_dataset('jet', data=jets[val_nevt:])
        dset = fh5.create_dataset('pid', data=y[val_nevt:])


if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=100, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/QGCMS', help="Folder containing input files")

    (flags, args) = parser.parse_args()
        

    samples_path = flags.folder
    NPARTS = flags.npoints

    dataset = ef.mod.load(amount=1.,
                          cache_dir=flags.folder,
                          collection='CMS2011AJets', 
                          dataset='sim', subdatasets=None, validate_files=False,
                          store_pfcs=True, store_gens=False, verbose=0)
    
    print(dataset)
    preprocess(dataset,samples_path)

