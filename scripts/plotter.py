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
    elif flags.dataset == 'opt':
        test = utils.TopDataLoader(os.path.join(flags.folder,'Opt', 'test_ttbar.h5'))

    elif flags.dataset == 'tau':
        test = utils.TauDataLoader(os.path.join(flags.folder,'TAU', 'test_tau.h5'))
    elif flags.dataset == 'qg':
        test = utils.QGDataLoader(os.path.join(flags.folder,'QG', 'test_qg.h5'))
        
    elif flags.dataset == 'cms':
        test = utils.CMSQGDataLoader(os.path.join(flags.folder,'CMSQG', 'test_qgcms_pid.h5'))
        
    elif flags.dataset == 'jetnet150':
        test = utils.JetNetDataLoader(os.path.join(flags.folder,'JetNet','test_150.h5'),big=True)
        
    elif flags.dataset == 'eic':
        test = utils.EicPythiaDataLoader(os.path.join(flags.folder,'EIC_Pythia','val_eic.h5'))
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
    print('number of events',parts.shape[0])
    print("number of particles", parts.shape[1])
    print('particles mean',np.mean(parts,(0,1)))
    print('particles std',np.std(parts,(0,1)))
    
    print('jets mean',np.mean(jets,0))
    print('jets std',np.std(jets,0))
    for feat in range(len(test.jet_names)):
        flat = jets[:, feat]
        fig, gs, _ = plot_utils.HistRoutine({'{}'.format(flags.dataset): flat}, test.jet_names[feat], 'Normalized Events', plot_ratio=False, reference_name='{}'.format(flags.dataset))
        fig.savefig(f"{flags.plot_folder}/jets_{flags.dataset}_{feat}.pdf", bbox_inches='tight')

    print("Maximum number of particles",np.max(np.sum(parts[:, :, 0]!=0,1)))
    mask = parts[:, :, 0].reshape(-1) != 0

    for feat in range(len(test.part_names)):
        flat = parts[:, :, feat].reshape(-1)
        flat = flat[mask]
        fig, gs, _ = plot_utils.HistRoutine({'{}'.format(flags.dataset): flat}, test.part_names[feat], 'Normalized Events', plot_ratio=False, reference_name='{}'.format(flags.dataset))
        fig.savefig(f"{flags.plot_folder}/parts_{flags.dataset}_{feat}.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()
