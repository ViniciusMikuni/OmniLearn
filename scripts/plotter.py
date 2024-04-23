import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import sys, gc
import utils
import matplotlib.pyplot as plt
import plot_utils
plot_utils.SetStyle()

        
parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--dataset", type="string", default="top", help="Folder containing input files")
parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
parser.add_option("--plot_folder", type="string", default="../plots", help="Folder to save the outputs")
parser.add_option("--n_bins", type=int, default=50, help="Number of bins for the histograms")

(flags, args) = parser.parse_args()

if flags.dataset == 'top':
    test = utils.TopDataLoader(os.path.join(flags.folder,'TOP', 'test_ttbar.h5'))

elif flags.dataset == 'qg':
    test = utils.QGDataLoader(os.path.join(flags.folder,'QG', 'test_qg.h5'))

elif flags.dataset == 'cms':
    test = utils.CMSQGDataLoader(os.path.join(flags.folder,'CMSQG', 'test_qgcms_pid.h5'))

elif flags.dataset == 'jetnet150':
    test = utils.JetNetDataLoader(os.path.join(flags.folder,'JetNet','test_150.h5'),big=True)
    
elif flags.dataset == 'jetnet30':
    test = utils.JetNetDataLoader(os.path.join(flags.folder,'JetNet','test_30.h5'))

elif flags.dataset == 'jetclass':
    test = utils.JetClassDataLoader(os.path.join(flags.folder,'JetClass','val'))
    
elif flags.dataset == 'h1':
    test = utils.H1DataLoader(os.path.join(flags.folder,'H1','val.h5'))
    
elif flags.dataset == 'atlas':
    test = utils.H1DataLoader(os.path.join(flags.folder,'ATLASTOP','val_atlas.h5'))

elif flags.dataset == 'omnifold':
    test = utils.OmniDataLoader(os.path.join(flags.folder,'OmniFold','test_pythia.h5'))
elif flags.dataset == 'lhco':
    test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO', 'val_background_SB.h5'))



jets = []
parts = []

for i, f in enumerate(test.files):
    print(i)
    if i > 40: continue
    X,y = test.data_from_file(f)
    parts.append(X[0])
    jets.append(X[3])
    del X
    
parts = np.concatenate(parts)
jets = np.concatenate(jets)


for feat in range(parts.shape[-1]):
    flat = parts[:,:,feat].reshape(-1)
    flat=flat[flat!=0]
    print(feat)
    print(np.mean(flat))
    print(np.std(flat))


print('jets mean',np.mean(jets,0))
print('jets std',np.std(jets,0))


if flags.dataset=='lhco':
    fig,gs,_ = plot_utils.HistRoutine({'{}'.format(flags.dataset): y},
                                      'mjj','Normalized Events', plot_ratio=False,
                                      reference_name = '{}'.format(flags.dataset))
    # plt.hist(y, bins=flags.n_bins,density=True)
    fig.savefig("{}/mjj.pdf".format(flags.plot_folder))


for feat in range(jets.shape[-1]):
    jet_names = ['Jet p$_{T}$ [GeV]', 'Jet $\eta$', 'Jet Mass [GeV]','Multiplicity']
    fig,gs,_ = plot_utils.HistRoutine({'{}'.format(flags.dataset): jets[:,feat]},
                                      jet_names[feat],
                                      'Normalized Events', plot_ratio=False,
                                      reference_name = '{}'.format(flags.dataset))
    
    # fig = plt.figure(figsize=(9, 9))
    # plt.hist(jets[:,feat], bins=flags.n_bins,density=True)
    fig.savefig("{}/jets_{}_{}.pdf".format(flags.plot_folder,flags.dataset,feat),bbox_inches='tight',)


for feat in range(parts.shape[-1]):
    flat = parts[:,:,feat].reshape(-1)
    part_names = ['$\eta_{rel}$', '$\phi_rel$', 'log($1 - p_{Trel}$)','log($p_{T}$)','log($1 - E_{rel}$)','log($E$)','$\Delta$R']
    fig,gs,_ = plot_utils.HistRoutine({'{}'.format(flags.dataset): flat[flat!=0]},
                                      part_names[feat],
                                      'Normalized Events', plot_ratio=False,
                                      reference_name = '{}'.format(flags.dataset))
    
        
    fig.savefig("{}/parts_{}_{}.pdf".format(flags.plot_folder,flags.dataset,feat),bbox_inches='tight')
