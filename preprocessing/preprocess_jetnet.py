import h5py as h5
import os
import numpy as np
from optparse import OptionParser
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
labels30 = {
    'g.hdf5':0,
    'q.hdf5':1,
    't.hdf5':2,
    'w.hdf5':3,
    'z.hdf5':4,
}

labels150 = {
    'g150.hdf5':0,
    'q150.hdf5':1,
    't150.hdf5':2,
    'w150.hdf5':3,
    'z150.hdf5':4,
}

# labels150 = {
#     'g150.hdf5':3,
#     'q150.hdf5':0,
#     't150.hdf5':8,
#     'w150.hdf5':7,
#     'z150.hdf5':6,
# }


def Recenter(particles):
    
    px = particles[:,:,2]*np.cos(particles[:,:,1])
    py = particles[:,:,2]*np.sin(particles[:,:,1])
    pz = particles[:,:,2]*np.sinh(particles[:,:,0])

    jet_px = np.sum(px,1)
    jet_py = np.sum(py,1)
    jet_pz = np.sum(pz,1)
    
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)
    jet_phi = np.ma.arctan2(jet_py,jet_px).filled(0)
    jet_eta = np.ma.arcsinh(np.ma.divide(jet_pz,jet_pt).filled(0))

    particles[:,:,0]-= np.expand_dims(jet_eta,1)
    particles[:,:,1]-= np.expand_dims(jet_phi,1)


    return particles


def process(p,j):
    mask = p[:,:,2]!=0
    new_p = np.zeros(shape=(p.shape[0],p.shape[1],7))
    p_e = j[:,None,0]*p[:,:,2]*np.cosh(p[:,:,0] + j[:,None,1])
    j_e = np.sqrt(j[:,None,0]**2 + j[:,None,2]**2)*np.cosh(j[:,None,1])
    
    #angles
    new_p[:,:,0] = p[:,:,0]
    new_p[:,:,1] = p[:,:,1]
    new_p[:,:,2] = np.ma.log(1.0 - p[:,:,2]).filled(0)
    new_p[:,:,3] = np.ma.log(p[:,:,2]*j[:,None,0]).filled(0)
    new_p[:,:,4] = np.ma.log(1.0 - p_e/j_e).filled(0)
    new_p[:,:,5] = np.ma.log(p_e).filled(0)
    new_p[:,:,6] = np.hypot(p[:,:,0],p[:,:,1])

    return new_p*mask[:,:,None]
    
def preprocess(path,labels):

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
            ntotal = h5f['jet_features'][:].shape[0]
            
            p = h5f['particle_features'][:].astype(np.float32)
            j = h5f['jet_features'][:].astype(np.float32)
            #p = Recenter(p)

            p = process(p,j)
            pid = to_categorical(labels[label]*np.ones(shape=(j.shape[0],1)), num_classes=5)

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
        
    return train,val,test

if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/PET/', help="Folder containing input files")
    parser.add_option("--label", type="string", default='150', help="Which dataset to use")
    (flags, args) = parser.parse_args()

    if '150' in flags.label:
        label = labels150
    else:
        label = labels30
    
    train,val,test = preprocess(os.path.join(flags.folder, 'JetNet'),label)

    with h5.File('{}/train_{}.h5'.format(os.path.join(flags.folder, 'JetNet'),flags.label), "w") as fh5:
        dset = fh5.create_dataset('data', data=train['data'])
        dset = fh5.create_dataset('pid', data=train['pid'])
        dset = fh5.create_dataset('jet', data=train['jet'])


    with h5.File('{}/test_{}.h5'.format(os.path.join(flags.folder, 'JetNet'),flags.label), "w") as fh5:
        dset = fh5.create_dataset('data', data=test['data'])
        dset = fh5.create_dataset('pid', data=test['pid'])
        dset = fh5.create_dataset('jet', data=test['jet'])


    with h5.File('{}/val_{}.h5'.format(os.path.join(flags.folder, 'JetNet'),flags.label), "w") as fh5:
        dset = fh5.create_dataset('data', data=val['data'])
        dset = fh5.create_dataset('pid', data=val['pid'])
        dset = fh5.create_dataset('jet', data=val['jet'])
