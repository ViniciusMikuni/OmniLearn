import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()


def read_file(
        filepath,
        max_num_particles=150,
        particle_features=['part_deta', 'part_dphi','log_part_pt_rel','log_part_pt',
                           'log_part_e_rel','log_part_e','part_hyp',                           
                           'part_charge','part_isChargedHadron','part_isNeutralHadron',
                           'part_isPhoton','part_isElectron','part_isMuon',
                           #'part_tanh_d0','part_d0err','part_tanh_dz','part_dzerr',
                           ],
        jet_features=['jet_pt','jet_eta', 'jet_mass', 'jet_nparticles'],
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):
    """Loads a single file from the JetClass dataset.

    **Arguments**

    - **filepath** : _str_
        - Path to the ROOT data file.
    - **max_num_particles** : _int_
        - The maximum number of particles to load for each jet. 
        Jets with fewer particles will be zero-padded, 
        and jets with more particles will be truncated.
    - **particle_features** : _List[str]_
        - A list of particle-level features to be loaded. 
        The available particle-level features are:
            - part_px
            - part_py
            - part_pz
            - part_energy
            - part_pt
            - part_eta
            - part_phi
            - part_deta: np.where(jet_eta>0, part_eta-jet_p4, -(part_eta-jet_p4))
            - part_dphi: delta_phi(part_phi, jet_phi)
            - part_d0val
            - part_d0err
            - part_dzval
            - part_dzerr
            - part_charge
            - part_isChargedHadron
            - part_isNeutralHadron
            - part_isPhoton
            - part_isElectron
            - part_isMuon
    - **jet_features** : _List[str]_
        - A list of jet-level features to be loaded. 
        The available jet-level features are:
            - jet_pt
            - jet_eta
            - jet_phi
            - jet_energy
            - jet_nparticles
            - jet_sdmass
            - jet_tau1
            - jet_tau2
            - jet_tau3
            - jet_tau4
    - **labels** : _List[str]_
        - A list of truth labels to be loaded. 
        The available label names are:
            - label_QCD
            - label_Hbb
            - label_Hcc
            - label_Hgg
            - label_H4q
            - label_Hqql
            - label_Zqq
            - label_Wqq
            - label_Tbqq
            - label_Tbl

    **Returns**

    - x_particles(_3-d numpy.ndarray_), x_jets(_2-d numpy.ndarray_), y(_2-d numpy.ndarray_)
        - `x_particles`: a zero-padded numpy array of particle-level features 
                         in the shape `(num_jets, num_particle_features, max_num_particles)`.
        - `x_jets`: a numpy array of jet-level features
                    in the shape `(num_jets, num_jet_features)`.
        - `y`: a one-hot encoded numpy array of the truth lables
               in the shape `(num_jets, num_classes)`.
    """

    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

    table = uproot.open(filepath)['tree'].arrays()

    p4 = vector.zip({'px': table['part_px'],
                     'py': table['part_py'],
                     'pz': table['part_pz'],
                     'energy': table['part_energy']})
    
    table['part_pt'] = p4.pt
    table['log_part_pt'] = np.log(table['part_pt'])
    table['log_part_e'] = np.log(table['part_energy'])
    table['log_part_pt_rel'] = np.log(1e-6 + 1.0 - table['part_pt']/table['jet_pt'])
    table['log_part_e_rel'] = np.log(1e-6 + 1.0 - table['part_energy']/table['jet_energy'])
    table['part_hyp'] = np.hypot(table['part_deta'],table['part_dphi'])
    # table['part_tanh_d0'] = np.tanh(table['part_d0val'])
    # table['part_tanh_dz'] = np.tanh(table['part_dzval'])
    table['part_deta'] = table['part_deta']*np.sign(table['jet_eta'])

    j4 = vector.zip({'pt': table['jet_pt'],
                     'eta': table['jet_eta'],
                     'phi': table['jet_phi'],
                     'energy': table['jet_energy']})


    table['jet_mass'] = j4.mass

    

    x_particles = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=-1)
    x_jets = np.stack([ak.to_numpy(table[n]).astype('float32') for n in jet_features], axis=-1)
    y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=1)

    return x_particles, x_jets, y
