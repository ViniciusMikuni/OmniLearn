import numpy as np
import utils
import plot_utils
import argparse

plot_utils.SetStyle()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and plot data based on the given dataset and configuration.")
    parser.add_argument("--dataset", type=str, default="top", help="Folder containing input files")
    parser.add_argument("--folder", type=str, default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_argument("--plot_folder", type=str, default="../plots", help="Folder to save the outputs")
    parser.add_argument("--mode", type=str, default="all", help="Loss type to train the model: [all/classifier/generator]")
    parser.add_argument("--local", action='store_true', help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--simple", action='store_true', help="Use simplified head model")
    parser.add_argument("--layer_scale", action='store_true', help="Use layer scale in the residual connections")
    return parser.parse_args()


def compute_means(input_array, M):
    # Ensure M is not zero to avoid division by zero
    if M <= 0:
        raise ValueError("M must be a positive integer")

    # Calculate the number of full chunks
    N = len(input_array)
    num_full_chunks = N // M

    # Initialize the result list
    result = []

    # Compute the mean for each full chunk
    for i in range(num_full_chunks):
        chunk_mean = np.mean(input_array[i * M:(i + 1) * M])
        result.append(chunk_mean)

    # Handle the last chunk if there are remaining elements that do not make up a full chunk
    remaining_elements = N % M
    if remaining_elements != 0:
        last_chunk_mean = np.mean(input_array[-remaining_elements:])
        result.append(last_chunk_mean)

    return np.array(result)


def load_and_plot_history(flags):

    baseline_file = utils.get_model_name(flags,fine_tune=False)
    ft_file = utils.get_model_name(flags,fine_tune=True)
    if flags.dataset == 'omnifold':
        baseline_file = f'{flags.folder}/histories/OmniFold_baseline_iter0_step1.pkl'
        ft_file = f'{flags.folder}/histories/OmniFold_fine_tune_iter0_step1.pkl'
            
    history_baseline = utils.load_pickle(flags.folder, baseline_file)
    history_ft = utils.load_pickle(flags.folder, ft_file)

    if flags.mode == 'generator':        
        loss_key = 'val_part'
        nchunk = 10        
    else:
        loss_key = 'val_loss'
        nchunk = 1
        
    plot_data = {
        f'{flags.dataset}_fine_tune': compute_means(history_ft[loss_key][:],nchunk),
        flags.dataset: compute_means(history_baseline[loss_key][:],nchunk),
    }
    fig, ax = plot_utils.PlotRoutine(plot_data, xlabel='Epochs' if loss_key == 'val_loss' else 'Epochs x 10', ylabel='Validation Loss', plot_min=True)
    fig.savefig(f"{flags.plot_folder}/loss_{flags.dataset}_{flags.mode}.pdf", bbox_inches='tight')

def main():
    flags = parse_arguments()
    load_and_plot_history(flags)

if __name__ == "__main__":
    main()
