import numpy as np
import os
import tensorflow as tf
import argparse
import logging

from omnifold import OmniFold, Classifier
import utils, plot_utils
import horovod.tensorflow.keras as hvd

def load_model(mc,flags, checkpoint_name):
    num_iter = 0 if flags.reco else flags.num_iter
    model_path = f"{flags.folder}/checkpoints/OmniFold_{checkpoint_name}_iter{num_iter}_step{'1' if flags.reco else '2'}.weights.h5"
    if hvd.rank() == 0:
        logging.info(f"Loading model {model_path}")

    model = Classifier(num_feat=mc.num_feat,
                       num_jet=mc.num_jet,
                       num_classes=mc.num_classes,
                       local = flags.local,
                       num_layers = flags.num_layers, 
                       drop_probability = flags.drop_probability,
                       simple = flags.simple, layer_scale = flags.layer_scale,
                       talking_head = flags.talking_head,
                       mode = 'classifier',
                       class_activation=None,
                       )
        
    model.load_weights(model_path)
    return model

def get_name_and_binning(nbins):
    binning = [
        np.linspace(0,75,nbins),
        np.linspace(0,0.6,nbins),
        np.linspace(0,80,80),
        np.linspace(-14,-2,nbins),
        np.linspace(0.0,0.5,nbins),
        np.linspace(0.0,1.2,nbins),
    ]

    name = ["Jet Mass [GeV]","Jet Width", "$n_{constituents}$",r"$ln\rho$","$z_g$",r"$\tau_{21}$"]
    return name, binning

def reweight_samples(inputs,model,mfold, batch_size=500):    
    return hvd.allgather(mfold.reweight(inputs,model,batch_size=batch_size)).numpy()

def calculate_triangle_distance(feed_dict, weights, binning, alternative_name,reference_name='herwig', ntrials=100):
    w = np.abs(binning[1] - binning[0])
    x, _ = np.histogram(feed_dict[reference_name], weights=weights[reference_name], bins=binning)
    x2,_ = np.histogram(feed_dict[reference_name], weights=weights[reference_name]**2, bins=binning)
    
    x_norm = np.sum(x)*w
    y, _ = np.histogram(feed_dict[alternative_name], weights=weights[alternative_name], bins=binning)
    y2, _ = np.histogram(feed_dict[alternative_name], weights=weights[alternative_name]**2, bins=binning)
    y_norm = np.sum(y)*w

    dist = sum(0.5 * w*(x[ib] / x_norm - y[ib] / y_norm) ** 2 / (x[ib] / x_norm + y[ib] / y_norm)
               if x[ib]+ y[ib] > 0 else 0.0 for ib in range(len(x)))

    x_plus = x + np.sqrt(x2)
    x_minus = x - np.sqrt(x2)
    y_plus = y + np.sqrt(y2)
    y_minus = y - np.sqrt(y2)

    results = []
    for trial in range(ntrials):
        x_ = np.random.uniform(low=x_minus, high=x_plus)
        y_ = np.random.uniform(low=y_minus, high=y_plus)
        d_ = sum(0.5 * w*(x_[ib] / x_norm - y_[ib] / y_norm) ** 2 / (x_[ib] / x_norm + y_[ib] / y_norm)
                 if x_[ib] + y_[ib] > 0 else 0.0 for ib in range(len(x)))
        results.append(d_)
    return dist * 1e3, np.std(results) * 1e3

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the OmniFold training and plotting routine.")
    parser.add_argument("--folder", type=str, default="/pscratch/sd/v/vmikuni/PET", help="Folder containing input files")
    parser.add_argument("--dataset", type=str, default="omnifold", help="Dataset to use")
    parser.add_argument("--mode", type=str, default="classifier", help="Loss type to train the model")
    parser.add_argument("--plot_folder", default="../plots", help="Folder to store plots")
    parser.add_argument("--num_iter", type=int, default=5, help="Iteration to load")
    parser.add_argument("--num_bins", type=int, default=50, help="Number of bins per histogram")
    parser.add_argument("--reco", action="store_true", help="Plot reco level results")
    parser.add_argument("--local", action='store_true', help="Use local embedding")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_argument("--simple", action='store_true', help="Use simplified head model")
    parser.add_argument("--talking_head", action='store_true', help="Use talking head attention")
    parser.add_argument("--layer_scale", action='store_true', help="Use layer scale in the residual connections")

    args = parser.parse_args()
    return args

def main():
    logging.basicConfig(level=logging.INFO)
    utils.setup_gpus()    
    plot_utils.SetStyle()
    flags = parse_arguments()

    mc = utils.OmniDataLoader(os.path.join(flags.folder,'OmniFold','train_pythia.h5'),flags.batch,
                              hvd.rank(),hvd.size())

    data = utils.OmniDataLoader(os.path.join(flags.folder,'OmniFold','train_herwig.h5'),flags.batch,
                                hvd.rank(),hvd.size())

    if flags.reco:
        mc_inputs = mc.reco
        mc_features = mc.high_level_reco
        data_features = data.high_level_reco
    else:
        mc_inputs = mc.gen
        mc_features = mc.high_level_gen
        data_features = data.high_level_gen

    mfold = OmniFold(version = 'baseline',
                     num_iter = flags.num_iter,
                     checkpoint_folder = flags.folder)

    
    model_baseline = load_model(mc,flags, 'baseline')
    unfolded_weights_baseline = reweight_samples(mc_inputs,model_baseline,mfold,batch_size=flags.batch)

    model_finetune = load_model(mc,flags, 'fine_tune')
    unfolded_weights_finetune = reweight_samples(mc_inputs,model_finetune,mfold,batch_size=flags.batch)

    mc_features = hvd.allgather(mc_features).numpy()
    data_features = hvd.allgather(tf.constant(data_features)).numpy()

    name,binning = get_name_and_binning(flags.num_bins)
    

    if hvd.rank() == 0:
        unfolded_name = 'pythia_reweighted' if flags.reco else 'pythia_unfolded'
        for feature in range(mc_features.shape[-1]):
            feed_dict = {
                unfolded_name + '_fine_tune': mc_features[:, feature],
                unfolded_name + '_baseline': mc_features[:, feature],
                'pythia': mc_features[:, feature],
                'herwig': data_features[:, feature],
            }
            weights = {
                unfolded_name + '_fine_tune': unfolded_weights_finetune,
                unfolded_name + '_baseline': unfolded_weights_baseline,
                'pythia': np.ones(mc_features.shape[0]),
                'herwig': np.ones(data_features.shape[0]),        
            }
            fig,ax,_ = plot_utils.HistRoutine(feed_dict,
                                              xlabel=name[feature],
                                              binning=binning[feature],
                                              weights = weights,
                                              label_loc='upper left',
                                              plot_ratio=True,
                                              reference_name='herwig',
                                              )
            fig.savefig('{}/omnifold_iter_{}_feat_{}.pdf'.format(flags.plot_folder,flags.num_iter,feature))
            
            d,derr = calculate_triangle_distance(feed_dict,weights,binning[feature],
                                                 alternative_name=unfolded_name+'_fine_tune')
            print("OmniLearn feat {}: {} +- {}".format(name[feature],d,derr))
            
            d,derr = calculate_triangle_distance(feed_dict,weights,binning[feature],
                                                 alternative_name=unfolded_name+'_baseline')
            print("Baseline feat {}: {} +- {}".format(name[feature],d,derr))
            

if __name__ == '__main__':
    main()

