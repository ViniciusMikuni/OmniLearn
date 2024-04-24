import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import plot_utils

plot_utils.SetStyle()


def parse_options():
    """Parse command line options."""
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Process some integers.")
    parser.add_argument("--dataset", type=str, default="top", help="Folder containing input files")
    parser.add_argument("--folder", type=str, default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_argument("--plot_folder", type=str, default="../plots", help="Folder to save the outputs")
    parser.add_argument("--n_bins", type=int, default=50, help="Number of bins for the histograms")
    return parser.parse_args()

def load_data(flags):
    """Load data based on dataset using specified data loaders and file naming conventions."""
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
        test = utils.JetClassDataLoader(os.path.join(flags.folder,'JetClass','test'))
        
    elif flags.dataset == 'h1':
        test = utils.H1DataLoader(os.path.join(flags.folder,'H1','val.h5'))
        
    elif flags.dataset == 'atlas':
        test = utils.H1DataLoader(os.path.join(flags.folder,'ATLASTOP','val_atlas.h5'))
        
    elif flags.dataset == 'omnifold':
        test = utils.OmniDataLoader(os.path.join(flags.folder,'OmniFold','test_pythia.h5'))
    elif flags.dataset == 'lhco':
        test = utils.LHCODataLoader(os.path.join(flags.folder,'LHCO', 'val_background_SB.h5'))        
    else:
        raise ValueError("Unknown dataset specified or file name not provided")

    return test

def process_particles(test):
    """Process particles and jets from the test dataset."""
    parts, jets = [], []
    for i, file_name in enumerate(test.files):
        if i > 40:
            break
        X, y = test.data_from_file(file_name)
        parts.append(X[0])
        jets.append(X[3])
        del X
    return np.concatenate(parts), np.concatenate(jets)

def main():
    flags = parse_options()
    plot_utils.SetStyle()
    
    test = load_data(flags)
    parts, jets = process_particles(test)

    print('particles mean',np.mean(parts,(0,1)))
    print('particles std',np.std(parts,(0,1)))
    
    print('jets mean',np.mean(jets,0))
    print('jets std',np.std(jets,0))
    
    part_names = ['$\eta_{rel}$', '$\phi_{rel}$', 'log($1 - p_{Trel}$)','log($p_{T}$)','log($1 - E_{rel}$)','log($E$)','$\Delta$R']
    jet_names = ['Jet p$_{T}$ [GeV]', 'Jet $\eta$', 'Jet Mass [GeV]','Multiplicity']
    
    for feat in range(len(jet_names)):
        flat = jets[:, feat]
        fig, gs, _ = plot_utils.HistRoutine({'{}'.format(flags.dataset): flat}, jet_names[feat], 'Normalized Events', plot_ratio=False, reference_name='{}'.format(flags.dataset))
        fig.savefig(f"{flags.plot_folder}/jets_{flags.dataset}_{feat}.pdf", bbox_inches='tight')
    
    for feat in range(len(part_names)):
        flat = parts[:, :, feat].reshape(-1)
        flat = flat[flat != 0]
        fig, gs, _ = plot_utils.HistRoutine({'{}'.format(flags.dataset): flat}, part_names[feat], 'Normalized Events', plot_ratio=False, reference_name='{}'.format(flags.dataset))
        fig.savefig(f"{flags.plot_folder}/parts_{flags.dataset}_{feat}.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()
