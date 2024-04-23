import os
import uproot3 as uproot
import pandas as pd
import numpy as np
import awkward
from optparse import OptionParser
from sklearn.utils import shuffle
import h5py

def find_files_with_string(directory, string):
    matching_files = []
    for filename in os.listdir(directory):
        if string in filename:
            matching_files.append(filename)
    return matching_files


def convert_to_np(file_list,base_path,name,
                  max_part = 30,
                  nevts=30000000):
    data_dict = {
        'jet':[],
        'data':[],
    }
    

    mask_list = ['y','ptmiss','Empz'] #Variables used only to determine the selection but not used during unfolding
    jet_list = ['jet_pt','jet_eta','jet_phi','jet_e','jet_mass']
    particle_list = ['jet_part_pt','jet_part_eta','jet_part_phi','jet_part_e','jet_part_charge','jet_part_idx']

    nevt = 0
    maxevt = int(3.5e6)
    for ifile,f in enumerate(file_list):
        print("evaluating file {}".format(f))
        if nevt >= maxevt:break
        print("Currently keeping {} events".format(nevt))
        try:
            tmp_file = uproot.open(os.path.join(base_path,f))['{}/minitree'.format(name)]
        except:
            'No TTree found, skipping'
            continue

        print("loaded file")

        mask_evt = (tmp_file['Q2'].array()[:] > 150)
        data_dict['jet'].append(np.stack([tmp_file[feat].array()[mask_evt].pad(1,clip=True).fillna(0).regular() for feat in jet_list],-1))        
        data_dict['data'].append(np.stack([tmp_file[feat].array()[mask_evt].pad(max_part,clip=True).fillna(0).regular() for feat in particle_list],-1))
        data_dict['jet'][-1] = np.squeeze(data_dict['jet'][-1])
        
        mask_reco = np.stack([tmp_file[feat].array()[mask_evt] for feat in mask_list],-1)
        print("Number of events: {}".format(mask_reco.shape[0]))

        # 0.08 < y < 0.7, ptmiss < 10, 45 < empz < 65
        pass_reco = (mask_reco[:,0] > 0.08) & (mask_reco[:,0] < 0.7) & (mask_reco[:,1]<10.0) & (mask_reco[:,2] > 45.) & (mask_reco[:,2] < 65) 

        data_dict['jet'][-1] = data_dict['jet'][-1][pass_reco]
        data_dict['data'][-1] = data_dict['data'][-1][pass_reco]
        del mask_reco, pass_reco
        
        #jet pt > 10 GeV and -1.5 < jet eta < 2.75
        pass_reco = (data_dict['jet'][-1][:,0] > 10) & (data_dict['jet'][-1][:,1] > -1.5) & (data_dict['jet'][-1][:,1] < 2.75)
        data_dict['jet'][-1] = data_dict['jet'][-1][pass_reco]
        data_dict['data'][-1] = data_dict['data'][-1][pass_reco]

        nevt += data_dict['jet'][-1].shape[0]
        
        # -1.5 < part eta < 2.75 and matched to leading jet
        mask_part = (data_dict['data'][-1][:,:,1] > -1.5) & (data_dict['data'][-1][:,:,1] < 2.75) & (data_dict['data'][-1][:,:,-1] == 0)
        data_dict['data'][-1] *= mask_part[:,:,None]
        mask_evt = np.sum(data_dict['data'][-1][:,:,0],1) > 0
        
        del mask_part
        print("Rejecting {}".format(1.0 - 1.0*np.sum(mask_evt)/mask_evt.shape[0]))
        data_dict['data'][-1] = data_dict['data'][-1][mask_evt]
        data_dict['jet'][-1] = data_dict['jet'][-1][mask_evt]


                
    data_dict['data'] = np.concatenate(data_dict['data'])[:maxevt]
    data_dict['jet'] = np.concatenate(data_dict['jet'])[:maxevt]
    
    
    # Make sure reco particles that do not pass reco cuts are indeed zero padded
    # order = np.argsort(-data_dict['data'][:,:,0],1)
    # data_dict['data'] = np.take_along_axis(data_dict['data'],order[:,:,None],1)
    del tmp_file
    return data_dict


def make_np_entries(particles,jets):
    mask = particles[:,:,0]>0
    NFEAT=8
    points = np.zeros((particles.shape[0],particles.shape[1],NFEAT))

    points[:,:,0] = particles[:,:,1]
    points[:,:,1] = particles[:,:,2]
    points[:,:,2] = np.ma.log(1.0 - particles[:,:,0]).filled(0)
    points[:,:,3] = np.ma.log(particles[:,:,0]*jets[:,0,None]).filled(0)
    
    points[:,:,4] = np.ma.log(1.0 - particles[:,:,3]).filled(0)    
    points[:,:,5] = np.ma.log(particles[:,:,3]*jets[:,3,None]).filled(0)
    points[:,:,6] = np.hypot(points[:,:,0],points[:,:,1])
    points[:,:,7] = particles[:,:,4]
    
    points*=mask[:,:,None]
    jets = np.concatenate([jets,np.sum(mask,-1)[:,None]],-1)
    #delete phi and energy
    jets = np.delete(jets,[2,3],axis=1)
    return points,jets

if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=30, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/PET/H1', help="Folder containing input files")

    (flags, args) = parser.parse_args()


    file_list = find_files_with_string(flags.folder+'/out_ep0607', 'Django_Eplus0607_')
    data_django = convert_to_np(file_list,flags.folder+'/out_ep0607',name='Django',max_part=flags.npoints)
    dj_p,dj_j = make_np_entries(data_django['data'],data_django['jet'])
        
    assert np.any(np.isnan(dj_p)) == False, "ERROR: NAN in particles"
    assert np.any(np.isnan(dj_j)) == False, "ERROR: NAN in jets"


    file_list = find_files_with_string(flags.folder+'/out_ep0607', 'Rapgap_Eplus0607_')
    data_rapgap = convert_to_np(file_list,flags.folder+'/out_ep0607',name='Rapgap',max_part=flags.npoints)
    rp_p,rp_j = make_np_entries(data_rapgap['data'],data_rapgap['jet'])
        
    assert np.any(np.isnan(rp_p)) == False, "ERROR: NAN in particles"
    assert np.any(np.isnan(rp_j)) == False, "ERROR: NAN in jets"

    particles, jets, pid = shuffle(np.concatenate([rp_p,dj_p],0),
                                   np.concatenate([rp_j,dj_j],0),
                                   np.concatenate([np.ones(rp_j.shape[0]),np.zeros(dj_j.shape[0])],0),
                                   )


    ntrain = int(2.5e6)
    nval = int(3.0e6)
    with h5py.File('{}/train.h5'.format(flags.folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=particles[:ntrain])
        dset = fh5.create_dataset('jet', data=jets[:ntrain])
        dset = fh5.create_dataset('pid', data=pid[:ntrain])

    with h5py.File('{}/val.h5'.format(flags.folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=particles[ntrain:nval])
        dset = fh5.create_dataset('jet', data=jets[ntrain:nval])
        dset = fh5.create_dataset('pid', data=pid[ntrain:nval])

    with h5py.File('{}/test.h5'.format(flags.folder), "w") as fh5:
        dset = fh5.create_dataset('data', data=particles[nval:])
        dset = fh5.create_dataset('jet', data=jets[nval:])
        dset = fh5.create_dataset('pid', data=pid[nval:])

    
