import os
import numpy as np
import h5py
from sklearn.model_selection import KFold
from dataloader import read_file
import gc
from optparse import OptionParser

def process_and_save(folder_path, n_splits,name):
    # List all files in the folder
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    output_dir = os.path.join(folder_path, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split the files into n_splits
    kf = KFold(n_splits=n_splits)

    for i, (_, split_indices) in enumerate(kf.split(all_files)):
        print("Running Fold {}".format(i))
        # Initialize lists to store concatenated data
        concat_X = []
        concat_y = []
        concat_j = []
        # Process each file in the current split
        for idx in split_indices:
            file_path = all_files[idx]
            X,j, y = read_file(file_path)

            concat_X.append(X)
            concat_y.append(y)
            concat_j.append(j)

        # Concatenate all X and y
        final_X = np.concatenate(concat_X, axis=0)
        final_y = np.concatenate(concat_y, axis=0)
        final_j = np.concatenate(concat_j, axis=0)

        # ----------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------        
        
        
        # Save to h5py file
        with h5py.File('{}/{}/JetClass_{}.h5'.format(folder_path,name,i), 'w') as h5f:
            h5f.create_dataset('data', data=final_X)
            h5f.create_dataset('jet', data=final_j)
            h5f.create_dataset('pid', data=final_y)
        del final_X, final_y, concat_X, concat_y
        gc.collect()

if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--sample", type="string", default='train', help="Input file name")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/PET/JetClass', help="Folder containing downloaded files")

    (flags, args) = parser.parse_args()
    
if flags.sample == 'train':
    process_and_save(os.path.join(flags.folder,'train_100M'), n_splits=1000,name='train')
if flags.sample == 'test':
    process_and_save(os.path.join(flags.folder,'test_20M'), n_splits=200,name='test')
if flags.sample == 'val':
    process_and_save(os.path.join(flags.folder,'val_5M'), n_splits=50,name='val')





