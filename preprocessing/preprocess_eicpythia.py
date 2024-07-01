import h5py as h5
import os
import sys
import numpy as np
from optparse import OptionParser
# from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from sklearn.utils import shuffle

'''particles = [pT, eta, phi, PID, z]'''
'''NO JET dateset. These are all particles EIC events'''

PID_INDEX = 3

labels = {
    'Pythia_26812400_10100000.h5'
}



def process(p):
    '''particles features: [pT, eta, phi, PID, z]'''

    mask = p[:,1:,0]!=0.0  # mask is from pT==0
    new_p = np.zeros(shape=(p.shape[0],p.shape[1]-1,13)) 
    
    #Modify the scattered electron pT for taking log later    
    print("Line 55, before feature shuffle\n", p[1,:3])
    new_p[:,:,0] = p[:,1:,1] + p[:,0,1,None]                             # eta
    new_p[:,:,1] = p[:,1:,2]                             # phi
    new_p[:,:,2] = np.ma.log(np.ma.divide(p[:,1:,0],p[:,0,0,None]).filled(0)).filled(0)  # pT
    new_p[:,:,3] = np.sign(p[:,1:,PID_INDEX])
    #hardcoded pids    
    new_p[:,:,4] = np.abs(p[:,1:,PID_INDEX]) == 2212. #proton
    new_p[:,:,5] = np.abs(p[:,1:,PID_INDEX]) == 2112. #neutron
    new_p[:,:,6] = np.abs(p[:,1:,PID_INDEX]) == 321. #kaon+
    new_p[:,:,7] = np.abs(p[:,1:,PID_INDEX]) == 211. #pi +
    new_p[:,:,8] = ((np.abs(p[:,1:,PID_INDEX]) == 16.) | (np.abs(p[:,1:,PID_INDEX]) == 14.) | (np.abs(p[:,1:,PID_INDEX]) == 12.)) # neutrinos
    new_p[:,:,9] = np.abs(p[:,1:,PID_INDEX]) == 13. #muon
    new_p[:,:,10] = np.abs(p[:,1:,PID_INDEX]) == 11. #electron
    new_p[:,:,11] = np.abs(p[:,1:,PID_INDEX]) == 22. #photon
    new_p[:,:,12] = np.abs(p[:,1:,PID_INDEX]) == 130. # k0l
    
    
    print("\n\nLine 60, AFTER feature shuffle\n", new_p[1,:3])
    # the main DataLoader class calculates disttanc/s, expecting eta and phi at 0 and 1

    new_p = new_p*mask[:,:,None]
    print("\n\nLine 84, after MASK\n", new_p[1,:3])
    # print(new_p[:2])
    return new_p, p[:,0,:2]
    
def preprocess(path, labels, nevent_max=-1, npart_max=-1):

    train = {
        'data':[],
        'jet':[],
        'pid':[],
    }

    test = {
        'data':[],
        'jet':[],
        'pid':[],
    }

    val = {
        'data':[],
        'jet':[],
        'pid':[],
    }

    for label in labels:        
        with h5.File(os.path.join(path,label),"r") as h5f:

            ntotal = h5f['particles'][:nevent_max].shape[0]

            p = h5f['particles'][:nevent_max,:npart_max].astype(np.float32)
            print(np.shape(p))

            p,j = process(p)  # applies mask, saves real pid, shuffles feature indecies for training
            j = np.concatenate([j,np.count_nonzero(p[:,:,2], axis=-1, keepdims=True)],1)  #pT moved to index 2
            # For Pythia EIC, there are no jets, we generate particles 
            # for the event. Jet here will be event properties,
            # Most importantly multiplicity


            pid = np.ones(np.shape(p[: , :, PID_INDEX]))  
            # for EIC pythia, PID is a particle feature
            # not a feature to be conditioned on for generation,
            # PID is in the particle data structure, 'p'. pid here is dummy


            train['data'].append(p[:int(0.63*ntotal)])
            train['jet'].append(j[:int(0.63*ntotal)])
            train['pid'].append(pid[:int(0.63*ntotal)])

            val['data'].append(p[int(0.63*ntotal):int(0.7*ntotal)])
            val['jet'].append(j[int(0.63*ntotal):int(0.7*ntotal)])
            val['pid'].append(pid[int(0.63*ntotal):int(0.7*ntotal)])

            test['data'].append(p[int(0.7*ntotal):])
            test['jet'].append(j[int(0.7*ntotal):])
            test['pid'].append(pid[int(0.7*ntotal):])

    for key in train:        
        train[key] = np.concatenate(train[key],0)        
    for key in test:
        test[key] = np.concatenate(test[key],0)
    for key in val:
        val[key] = np.concatenate(val[key],0)


    for d in [train,test,val]:
        d['data'],d['jet'],d['pid'] = shuffle(d['data'],d['jet'],d['pid'])
        
    return train, val, test

if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default='/global/cfs/cdirs/m4662', help="Folder containing input files")
    parser.add_option("--nevent_max", type="int", default='10_000_000', help="max number of events")
    parser.add_option("--npart_max", type="int", default='50', help="max number of particses per event")
    (flags, args) = parser.parse_args()

    
    train, val, test = preprocess(os.path.join(flags.folder, 'diffusion'),labels, flags.nevent_max, flags.npart_max)

    with h5.File('{}/train_{}.h5'.format(os.path.join(flags.folder, 'diffusion'),"eic"), "w") as fh5:
        dset = fh5.create_dataset('data', data=train['data'])
        dset = fh5.create_dataset('pid', data=train['pid'])
        dset = fh5.create_dataset('jet', data=train['jet'])


    with h5.File('{}/test_{}.h5'.format(os.path.join(flags.folder, 'diffusion'),"eic"), "w") as fh5:
        dset = fh5.create_dataset('data', data=test['data'])
        dset = fh5.create_dataset('pid', data=test['pid'])
        dset = fh5.create_dataset('jet', data=test['jet'])


    with h5.File('{}/val_{}.h5'.format(os.path.join(flags.folder, 'diffusion'),"eic"), "w") as fh5:
        dset = fh5.create_dataset('data', data=val['data'])
        dset = fh5.create_dataset('pid', data=val['pid'])
        dset = fh5.create_dataset('jet', data=val['jet'])

