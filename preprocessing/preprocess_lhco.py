import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import sys
import utils

parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")


(flags, args) = parser.parse_args()

bkg = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','processed_data_background_rel.h5'))
signal = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO','processed_data_signal_rel.h5'))

samples = {
    'background':bkg,
    'signal':signal,
    }

regions = {
    'SR':True,
    'SB':False,
    }

train_frac = 0.8
for sample in samples:
    for region in regions:
        if sample == 'signal' and region == 'SB': continue
        mask = samples[sample].get_mjj_mask(samples[sample].raw_y,use_SR=regions[region],mjjmin=2300,mjjmax=5000) & (np.abs(samples[sample].jet[:,0,-1])>1.0) & (np.abs(samples[sample].jet[:,1,-1])>1.0)
        data = samples[sample].X[mask]
        jet = samples[sample].jet[mask]
        mass = samples[sample].raw_y[mask]

        nevts = jet.shape[0]
        with h5.File('{}/LHCO/train_{}_{}.h5'.format(flags.folder,sample,region), "w") as fh5:
            dset = fh5.create_dataset('data', data=data[:int(train_frac*nevts)])
            dset = fh5.create_dataset('jet', data=jet[:int(train_frac*nevts)])
            dset = fh5.create_dataset('pid', data=mass[:int(train_frac*nevts)])

        with h5.File('{}/LHCO/val_{}_{}.h5'.format(flags.folder,sample,region), "w") as fh5:
            dset = fh5.create_dataset('data', data=data[int(train_frac*nevts):])
            dset = fh5.create_dataset('jet', data=jet[int(train_frac*nevts):])
            dset = fh5.create_dataset('pid', data=mass[int(train_frac*nevts):])


    
