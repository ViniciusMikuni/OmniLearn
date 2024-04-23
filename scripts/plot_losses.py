import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import sys, gc
import utils
import matplotlib.pyplot as plt
import pickle
import utils
import plot_utils
plot_utils.SetStyle()


def compute_means(input_array, M):
    N = len(input_array)
    if M >= N:
        raise ValueError("M must be less than N")
    
    interval = int(np.ceil(N / M))  # Calculate interval size, rounded up to ensure coverage of the entire array
    result = []
    
    for i in range(0, N, interval):
        # Compute the mean for each interval, ensuring we don't go out of bounds
        interval_mean = np.mean(input_array[i:i+interval])
        result.append(interval_mean)
    
    # If the result array is larger than M due to ceiling, trim the excess
    if len(result) > M:
        result = result[:M]
    
    return np.array(result)

parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--dataset", type="string", default="top", help="Folder containing input files")
parser.add_option("--folder", type="string", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
parser.add_option("--plot_folder", type="string", default="../plots", help="Folder to save the outputs")
parser.add_option("--mode", type="string", default="all", help="Loss type to train the model: available options are [all/classifier/generator]")
parser.add_option('--local', action='store_true', default=False,help='Use local embedding')
parser.add_option("--num_layers", type=int, default=8, help="Number of transformer layers")
parser.add_option('--simple', action='store_true', default=False,help='Use simplified head model')
parser.add_option('--layer_scale', action='store_true', default=False,help='Use layer scale in the residual connections')


(flags, args) = parser.parse_args()

baseline_file = utils.get_model_name(flags,fine_tune=False)
if flags.dataset == 'omnifold':
    baseline_file = '{}/histories/OmniFold_baseline_iter0_step1.pkl'.format(flags.folder)
ft_file = utils.get_model_name(flags,fine_tune=True)
if flags.dataset == 'omnifold':
    ft_file = '{}/histories/OmniFold_fine_tune_iter0_step1.pkl'.format(flags.folder)
history_baseline = utils.load_pickle(flags.folder,baseline_file)
history_ft = utils.load_pickle(flags.folder,ft_file)

loss_dict = {
    'omnifold':'val_loss',
    'jetnet30':'val_part',
    'jetnet150':'val_part',
    'lhco':'val_part',
    'top':'val_loss',
    'qg':'val_loss',
    'h1':'val_loss',
    'atlas':'val_loss',
    'atlas_small':'val_loss',
    'cms':'val_loss',
}

print(history_ft[loss_dict[flags.dataset]])
plot_dict = {
    '{}_fine_tune'.format(flags.dataset):history_ft[loss_dict[flags.dataset]],
    '{}'.format(flags.dataset):history_baseline[loss_dict[flags.dataset]],
    # '{}_fine_tune'.format(flags.dataset):compute_means(history_ft[loss_dict[flags.dataset]][:],10),
    # '{}'.format(flags.dataset):compute_means(history_baseline[loss_dict[flags.dataset]][:],10),
}

fig,ax = plot_utils.PlotRoutine(plot_dict,xlabel='Epochs',ylabel='Validation Loss',plot_min=True)
#ax.set_ylim([0.716,0.73]) #jetnet30
#ax.set_ylim([0.742,0.752]) #jetnet150
# ax.set_ylim([0.686,0.694]) #lhco
fig.savefig("{}/loss_{}_{}.pdf".format(flags.plot_folder,flags.dataset,flags.mode),bbox_inches='tight')
