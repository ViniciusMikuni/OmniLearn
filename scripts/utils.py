import numpy as np
import h5py as h5
from sklearn.utils import shuffle
import sys
import os
import tensorflow as tf
import gc
import random
import itertools
import pickle, copy
from scipy.stats import norm
import horovod.tensorflow.keras as hvd

def setup_gpus():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def get_model_name(flags,fine_tune=False,add_string=""):
    model_name = 'PET_{}_{}_{}_{}_{}_{}_{}{}.weights.h5'.format(
        flags.dataset,
        flags.num_layers,
        'local' if flags.local else 'nolocal',
        'layer_scale' if flags.layer_scale else 'nolayer_scale',
        'simple' if flags.simple else 'token',
        'fine_tune' if fine_tune else 'baseline',        
        flags.mode,
        add_string,
    )
    return model_name

def load_pickle(folder,f):
    file_name = os.path.join(folder,'histories',f.replace(".weights.h5",".pkl"))
    with open(file_name, 'rb') as file_pi:
        history_dict = pickle.load(file_pi)
    return history_dict

def revert_npart(npart, name='30'):
    # Reverse the preprocessing to recover the particle multiplicity
    stats = {'30': (29.03636, 2.7629626),
             '49': (21.66242333, 8.86935969),
             '150': (49.398304, 20.772636),
             '279': (57.28675, 29.41252836)}
    mean, std = stats[name]
    return np.round(npart * std + mean).astype(np.int32)


class DataLoader:
    """Base class for all data loaders with common preprocessing methods."""
    def __init__(self, path, batch_size=512, rank=0, size=1, **kwargs):

        self.path = path
        self.batch_size = batch_size
        self.rank = rank
        self.size = size

        self.mean_part = [0.0, 0.0, -0.0278,
                          1.8999407,-0.027,2.244736, 0.0,
                          0.0, 0.0,  0.0,  0.0,  0.0, 0.0]
        self.std_part = [0.215, 0.215,  0.070, 
                         1.2212526, 0.069,1.2334691,1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.mean_jet =  [ 6.18224920e+02, 0.0, 1.2064709e+02,3.94133173e+01]
        self.std_jet  = [106.71761,0.88998157,40.196922,15.096386]

        self.part_names = ['$\eta_{rel}$', '$\phi_{rel}$', 'log($1 - p_{Trel}$)','log($p_{T}$)','log($1 - E_{rel}$)','log($E$)','$\Delta$R']
        self.jet_names = ['Jet p$_{T}$ [GeV]', 'Jet $\eta$', 'Jet Mass [GeV]','Multiplicity']
        
    
    def pad(self,x,num_pad):
        return np.pad(x, pad_width=((0, 0), (0, 0), (0, num_pad)),
                      mode='constant', constant_values=0)

    def data_from_file(self,file_path, preprocess=False):
        with h5.File(file_path, 'r') as file:
            data_chunk = file['data'][:]
            mask_chunk = data_chunk[:, :, 2] != 0
            
            jet_chunk = file['jet'][:]
            label_chunk = file['pid'][:]

            if preprocess:
                data_chunk = self.preprocess(data_chunk, mask_chunk)
                data_chunk = self.pad(data_chunk,num_pad=self.num_pad)
                jet_chunk = self.preprocess_jet(jet_chunk)
                
            points_chunk = data_chunk[:, :, :2]
            
        return [data_chunk,points_chunk,mask_chunk,jet_chunk],label_chunk

    def make_eval_data(self):
        X = self.preprocess(self.X,self.mask).astype(np.float32)
        X = self.pad(X,num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).astype(np.float32)

        tf_zip = tf.data.Dataset.from_tensor_slices(
            {'input_features':X,
             'input_points':X[:,:,:2],
             'input_mask':self.mask.astype(np.float32),
             'input_jet':jet,
             'input_time':np.zeros((self.jet.shape[0],1)),})
                        
        return tf_zip.cache().batch(self.batch_size).prefetch(tf.data.AUTOTUNE), self.y

    def make_tfdata(self):
        X = self.preprocess(self.X,self.mask).astype(np.float32)
        X = self.pad(X,num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).astype(np.float32)
        tf_zip = tf.data.Dataset.from_tensor_slices(
            {'input_features':X,
             'input_points':X[:,:,:2],
             'input_mask':self.mask.astype(np.float32),
             'input_jet':jet})
        

        tf_y = tf.data.Dataset.from_tensor_slices(self.y)
        del self.X, self.y,  self.mask
        gc.collect()
        
        return tf.data.Dataset.zip((tf_zip,tf_y)).cache().shuffle(self.batch_size*100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None):
        # self.path = path

        self.X = h5.File(self.path,'r')['data'][rank:nevts:size]
        self.y = h5.File(self.path,'r')['pid'][rank:nevts:size]
        self.jet = h5.File(self.path,'r')['jet'][rank:nevts:size]
        self.mask = self.X[:,:,2]!=0

        # self.batch_size = batch_size
        self.nevts = h5.File(self.path,'r')['data'].shape[0] if nevts is None else nevts
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]


    def preprocess(self,x,mask):                
        num_feat = x.shape[-1]
        return mask[:,:, None]*(x[:,:,:num_feat]-self.mean_part[:num_feat])/self.std_part[:num_feat]

    def preprocess_jet(self,x):        
        return (x-self.mean_jet)/self.std_jet

    def revert_preprocess(self,x,mask):                
        num_feat = x.shape[-1]        
        new_part = mask[:,:, None]*(x[:,:,:num_feat]*self.std_part[:num_feat] + self.mean_part[:num_feat])
        #log pt rel and log e rel  should always be negative or 0
        new_part[:,:,2] = np.minimum(new_part[:,:,2],0.0)
        return  new_part

    def revert_preprocess_jet(self,x):

        new_x = self.std_jet*x+self.mean_jet
        #Convert multiplicity back into integers
        new_x[:,-1] = np.round(new_x[:,-1])
        new_x[:,-1] = np.clip(new_x[:,-1],1,self.num_part)
        return new_x


class EicPythiaDataLoader(DataLoader):
    '''based off jetnet. No jets, just events and particles'''

    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.mean_part = [-6.57722423e-01, -1.32635604e-04, -1.35429178,
                          0.0, 0.0, 0.0,0.0, 0.0, 0.0,0.0, 0.0, 0.0,0.0]
        self.std_part = [1.43289689, 0.95137615, 1.49257704,
                         1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0]
        self.mean_jet = [ 6.48229788, -2.52708796,  21.66242333]
        self.std_jet  = [2.82288916, 0.4437837,  8.86935969]
        
        self.part_names = ['$\eta_{rel}$', '$\phi_{rel}$', 'log($p_{Trel}$)',
                           'charge','is proton','is neutron','is kaon',
                           'is pion', 'is neutrino',
                           'is muon', 'is electron',
                           'is photon', 'is pi0']
        self.jet_names = ['electron $p_T$ [GeV]','electron $\eta$','Multiplicity']

            
        def add_noise(self,x,shape=None):
            #Add noise to the event multiplicity
            if shape is None:
                noise = np.random.uniform(-0.3,0.3,x.shape[0])
                x[:,-1]+=noise[:,None]
            else:
                noise = np.random.uniform(-0.3,0.3,shape)
                x[:,:,4:]+=noise[:,None]
            return x
            
        def preprocess_jet(self,x):
            #new_x = self.add_noise(copy.deepcopy(x))
            return (new_x-self.mean_jet)/self.std_jet

        def preprocess(self,x,mask):                
            num_feat = x.shape[-1]
            #new_x = self.add_noise(copy.deepcopy(x),x[:,:,4:].shape)
            return mask[:,:, None]*(new_x[:,:,:num_feat]-self.mean_part[:num_feat])/self.std_part[:num_feat]


        self.load_data(path, batch_size,rank,size)
        #the model is not conditioned
        self.y = np.zeros((self.X.shape[0],1))
        if rank ==0:
            print(f"Loaded dataset with {self.num_part} particles")
        self.num_feat = self.X.shape[2]
        self.num_classes = self.y.shape[1]
        self.num_pad = 0
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

    def revert_preprocess(self,x,mask):                
        num_feat = x.shape[-1]        
        new_part = mask[:,:, None]*(x[:,:,:num_feat]*self.std_part[:num_feat] + self.mean_part[:num_feat])
        #one hot encode the pids again
        max_indices = np.argmax(new_part[:,:,4:], axis=-1)
        pids = np.zeros_like(new_part[:,:,4:])
        pids[np.arange(new_part.shape[0])[:, None], np.arange(new_part.shape[1]), max_indices] = 1
        new_part[:,:,4:] = pids
        
        return  new_part



class JetNetDataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1,big=False):
        super().__init__(path, batch_size, rank, size)
        if big:
            self.mean_part = [0.0, 0.0, -0.0217,
                              1.895,-0.022, 2.13, 0.0,
                              0.0, 0.0,  0.0,  0.0,  0.0, 0.0]
            self.std_part = [0.115, 0.115, -0.054,  
                             1.549, 0.054,1.57,1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            
            self.mean_jet =  [1.0458962e+03, 3.6804923e-03, 9.4020386e+01, 4.9398304e+01]
            self.std_jet  = [123.23525 ,0.7678173 ,43.103817 ,20.772703 ]
        else:
            self.mean_part = [0.0, 0.0, -0.035,
                              2.791,-0.035, 3.03, 0.0,
                              0.0, 0.0,  0.0,  0.0,  0.0, 0.0]
            self.std_part = [0.09, 0.09,  0.067, 
                             1.241, 0.067,1.26,1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            
            self.mean_jet =  [1.0458962e+03, 3.6804923e-03, 9.4020386e+01, 2.9036360e+01]
            self.std_jet  = [123.23525 ,0.7678173 ,43.103817 ,2.76302]
            
        def add_noise(self,x):
            #Add noise to the jet multiplicity
            noise = np.random.uniform(-0.5,0.5,x.shape[0])
            x[:,-1]+=noise[:,None]
            return x
            
        def preprocess_jet(self,x):
            new_x = self.add_noise(copy.deepcopy(x))
            return (new_x-self.mean_jet)/self.std_jet


        self.load_data(path, batch_size,rank,size)
        self.big = big        
        self.num_pad = 6
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]



class LHCODataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1,
                 mjjmin=2300,
                 mjjmax=5000,nevts = -1):
        super().__init__(path, batch_size, rank, size)

        self.mean_part = [0.0, 0.0, -0.019,
                          1.83,-0.019, 2.068, 0.0,
                          0.0, 0.0,  0.0,  0.0,  0.0, 0.0]
        self.std_part = [0.26, 0.26,  0.066, 
                         1.452, 0.064,1.46,1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.mean_jet =  [ 1.28724651e+03, -4.81260266e-05, 0.0 , 2.05052711e+02, 5.72253125e+01]
        self.std_jet = [244.15460668 ,0.74111563, 1.0 ,151.10313677, 29.44343823]

        
        self.path = path
        if nevts <0:
            self.nevts = h5.File(self.path,'r')['jet'].shape[0]
        else:
            self.nevts = nevts
        self.size = size

        self.jet = h5.File(self.path,'r')['jet'][rank:int(self.nevts):size]
        self.X = h5.File(self.path,'r')['data'][rank:int(self.nevts):size]
        #Min pT cut 
        self.X = self.X*(self.X[:,:,:,3:4]>-0.0)

        if 'pid' in h5.File(self.path,'r'):
            self.raw_y = h5.File(self.path,'r')['pid'][rank:int(self.nevts):size]
        else:
            self.raw_y = self.get_dimass(self.jet)
            
        self.y = self.prep_mjj(self.raw_y)

        self.mask = self.X[:,:,:,2]!=0
        self.batch_size = batch_size
        
        self.num_part = self.X.shape[2]
        self.num_pad = 6
        self.num_feat = self.X.shape[3] + self.num_pad #missing inputs
        self.num_jet = self.jet.shape[2]
        self.num_classes = 1
        self.steps_per_epoch = None
        self.files = [path]
        self.label = np.zeros((self.y.shape[0],1))
        

    def LoadMjjFile(self,folder,file_name,use_SR,mjjmin=2300,mjjmax=5000):    
        with h5.File(os.path.join(folder,file_name),"r") as h5f:
            mjj = h5f['mjj'][:]

        mask = self.get_mjj_mask(mjj,use_SR,mjjmin,mjjmax)
        mjj = self.prep_mjj(mjj)
        return mjj[mask]
        
    def prep_mjj(self,mjj,mjjmin=2300,mjjmax=5000):
        new_mjj = (mjj - mjjmin)/(mjjmax - mjjmin)
        #new_mjj = (np.log(mjj) - np.log(mjjmin))/(np.log(mjjmax) - np.log(mjjmin))
        new_mjj = 2*new_mjj -1.0
        new_mjj = np.stack([new_mjj,np.ones_like(new_mjj)],-1)
        return new_mjj.astype(np.float32)

    def revert_mjj(self,mjj,mjjmin=2300,mjjmax=5000):
        x = (mjj[:,0] + 1.0)/2.0        
        x = x * ( mjjmax - mjjmin ) + mjjmin
        return x
        logmin = np.log(mjjmin)
        logmax = np.log(mjjmax)
        x = mjj * ( logmax - logmin ) + logmin
        return np.exp(x)
        
    def get_dimass(self,jets):
        jet_e = np.sqrt(jets[:,0,3]**2 + jets[:,0,0]**2*np.cosh(jets[:,0,1])**2)
        jet_e += np.sqrt(jets[:,1,3]**2 + jets[:,1,0]**2*np.cosh(jets[:,1,1])**2)
        jet_px = jets[:,0,0]*np.cos(jets[:,0,2]) + jets[:,1,0]*np.cos(jets[:,1,2])
        jet_py = jets[:,0,0]*np.sin(jets[:,0,2]) + jets[:,1,0]*np.sin(jets[:,1,2])
        jet_pz = jets[:,0,0]*np.sinh(jets[:,0,1]) + jets[:,1,0]*np.sinh(jets[:,1,1])
        mjj = np.sqrt(np.abs(jet_px**2 + jet_py**2 + jet_pz**2 - jet_e**2))
        return mjj
    
    def get_mjj_mask(self,mjj,use_SR,mjjmin,mjjmax):
        if use_SR:
            mask_region = (mjj>3300) & (mjj<3700)
        else:
            mask_region = ((mjj<3300) & (mjj>mjjmin)) | ((mjj>3700) & (mjj<mjjmax))
        return mask_region

    def pad(self,x,num_pad):
        return np.pad(x, pad_width=((0, 0), (0, 0), (0, 0), (0, num_pad)),
                      mode='constant', constant_values=0)

    def make_eval_data(self):
        X = self.preprocess(self.X,self.mask).astype(np.float32)
        X = self.pad(X,num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).astype(np.float32)

        tf_zip = tf.data.Dataset.from_tensor_slices(
            {'input_features':X,
             'input_points':X[:,:,:,:2],
             'input_mask':self.mask.astype(np.float32),
             'input_jet':jet,
             'input_mass': self.y[:,0],
             'input_time':np.zeros((self.jet.shape[0],2,1)),})
                        
        return tf_zip.cache().batch(self.batch_size).prefetch(tf.data.AUTOTUNE), self.label

    def add_noise(self,x):
        #Add noise to the jet multiplicity
        noise = np.random.uniform(-0.5,0.5,x.shape[0])
        x[:,:,-1]+=noise[:,None]
        return x
    
    def make_tfdata(self,classification=False):
        X = self.preprocess(self.X,self.mask).astype(np.float32)
        X = self.pad(X,num_pad=self.num_pad)
        jet = self.add_noise(self.jet)
        jet = self.preprocess_jet(jet).astype(np.float32)
        tf_zip = tf.data.Dataset.from_tensor_slices(
            {'input_features':X,
             'input_points':X[:,:,:,:2],
             'input_mask':self.mask.astype(np.float32),
             'input_mass':self.y[:,0],
             'input_jet':jet})
        
        if classification:
            tf_y = tf.data.Dataset.from_tensor_slices(np.concatenate([self.label,self.w],-1))
        else:
            tf_y = tf.data.Dataset.from_tensor_slices(self.y)
        del self.X, self.y,  self.mask
        gc.collect()
        
        return tf.data.Dataset.zip((tf_zip,tf_y)).cache().shuffle(self.batch_size*100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def combine(self,datasets,use_weights=False):        
        for dataset in datasets:
            self.nevts += dataset.nevts
            self.X = np.concatenate([self.X,dataset.X],0)
            self.mask = np.concatenate([self.mask,dataset.mask],0)
            self.jet = np.concatenate([self.jet,dataset.jet],0)
            self.label = np.concatenate([self.label,np.ones((dataset.y.shape[0],1))],0)
            self.y = np.concatenate([self.y,dataset.y],0)
            if use_weights:
                self.w = np.concatenate([self.w,np.ones((dataset.y.shape[0],1))],0)
        if use_weights:
            self.X,self.mask,self.jet,self.label,self.y,self.w = shuffle(
                self.X,self.mask,self.jet,self.label,self.y,self.w)
        else:
            self.X,self.mask,self.jet,self.label,self.y= shuffle(self.X,self.mask,self.jet,self.label,self.y)
        
        #Update parameters        
        #self.steps_per_epoch = int(self.nevts//self.size//self.batch_size)

    def data_from_file(self,file_path):
        with h5.File(file_path, 'r') as file:
            data_chunk = file['data'][:]
            N,J,P,F = data_chunk.shape
            mask_chunk = data_chunk[:, :, :,2] != 0  


            jet_chunk = file['jet'][:]
            label_chunk = self.get_dimass(jet_chunk)
                        
            data_chunk = self.preprocess(data_chunk, mask_chunk)
            data_chunk = self.pad(data_chunk,num_pad=self.num_pad)
            jet_chunk = self.preprocess_jet(jet_chunk)
            points_chunk = data_chunk[:, :, :,:2]            
            data_chunk = data_chunk.reshape(N*J,P,-1)
            jet_chunk = jet_chunk.reshape(N*J,-1)
            
        return [data_chunk,points_chunk,mask_chunk,jet_chunk],label_chunk


    def preprocess_jet(self,x):
        #Transform phi from uniform to gaussian
        new_x = copy.deepcopy(x)
        new_x[:,:,2] = norm.ppf(0.5*(1.0 + x[:,:,2]/np.pi))
        return (new_x-self.mean_jet)/self.std_jet

    def preprocess(self,x,mask):        
        num_feat = x.shape[-1]
        return mask[:,:,:, None]*(x[:,:,:,:num_feat]-self.mean_part[:num_feat])/self.std_part[:num_feat]

    def revert_preprocess(self,x,mask):                
        num_feat = x.shape[-1]

        new_part = mask[:,:,:, None]*(x[:,:,:,:num_feat]*self.std_part[:num_feat] + self.mean_part[:num_feat])
        #log pt rel should always be negative or 0
        new_part[:,:,:,2] = np.minimum(new_part[:,:,:,2],0.0)
        return  new_part
        
    def revert_preprocess_jet(self,x):
        new_x = self.std_jet*x+self.mean_jet
        #Recover phi
        new_x[:,:,2] = np.pi*(2*norm.cdf(new_x[:,:,2]) -1.0)
        new_x[:,:,2] = np.clip(new_x[:,:,2],-np.pi,np.pi)
        #Convert multiplicity back into integers
        new_x[:,:,-1] = np.round(new_x[:,:,-1])
        new_x[:,:,-1] = np.clip(new_x[:,:,-1],2,self.num_part)
        return new_x

class TopDataLoader(DataLoader):    
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size)
        self.num_pad = 6
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]


class ToyDataLoader(DataLoader):    
    def __init__(self, nevts,batch_size=512,rank=0,size=1):
        super().__init__(nevts,batch_size, rank, size)

        self.nevts = nevts
        self.X = np.concatenate([
            np.random.normal(loc = 0.0,scale=1.0,size=(self.nevts,15,13)),
            np.random.normal(loc = 1.0,scale=1.0,size=(self.nevts,15,13))])
        self.jet = np.concatenate([
            np.random.normal(loc = 0.0,scale=1.0,size=(self.nevts,4)),
            np.random.normal(loc = 1.0,scale=1.0,size=(self.nevts,4))])
        self.mask = self.X[:,:,2]!=0
        self.y = np.concatenate([np.ones((self.nevts)),np.zeros((self.nevts))])        
        self.num_part = self.X.shape[1]
        self.num_jet = self.jet.shape[1]

        
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        #one hot label
        self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = None

        

class TauDataLoader(DataLoader):    
    def __init__(self, path, batch_size=512,rank=0,size=1,nevts=None):
        super().__init__(path, batch_size, rank, size)

        self.mean_part = [ 0.0, 0.0, -4.68198519e-02,  2.20178221e-01,
                                -7.48168704e-02,  2.56480441e-01,  0.0,
                                0.0, 0.0,  0.0,  0.0,  0.0, 0.0]
        self.std_part =  [0.03927566, 0.04606768, 0.25982114,
                               0.82466037, 0.7541279,  0.86455974,1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.mean_jet = [6.16614813e+01, 2.05619964e-03, 3.52885518e+00, 4.28755680e+00]
        self.std_jet  = [34.22578952,  0.68952567,  4.54982729,  3.20547624]

        self.load_data(path, batch_size,rank,size,nevts = nevts)

        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

        
class AtlasDataLoader(DataLoader):    
    def __init__(self, path, batch_size=512,rank=0,size=1,is_small=False):
        super().__init__(path, batch_size, rank, size)
        self.mean_jet =  [1.73933684e+03, 4.94380870e-04, 2.21667582e+02, 5.52376512e+01]
        self.std_jet  = [9.75164004e+02, 8.31232765e-01, 2.03672420e+02, 2.51242747e+01]
        
        self.path = path
        if is_small:
            self.nevts = int(4e6)
        else:
            self.nevts = h5.File(self.path,'r')['data'].shape[0]
            
        self.X = h5.File(self.path,'r')['data'][rank:self.nevts:size]
        self.y = h5.File(self.path,'r')['pid'][rank:self.nevts:size]
        self.w = h5.File(self.path,'r')['weights'][rank:self.nevts:size]
        self.jet = h5.File(self.path,'r')['jet'][rank:self.nevts:size]
        self.mask = self.X[:,:,2]!=0

        self.batch_size = batch_size
        
        self.num_part = self.X.shape[1]
        self.num_pad = 6

        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        self.num_jet = self.jet.shape[1]
        self.num_classes = 1
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

    def make_tfdata(self):
        X = self.preprocess(self.X,self.mask).astype(np.float32)
        X = self.pad(X,num_pad=self.num_pad)
        jet = self.preprocess_jet(self.jet).astype(np.float32)
        tf_zip = tf.data.Dataset.from_tensor_slices(
            {'input_features':X,
             'input_points':X[:,:,:2],
             'input_mask':self.mask.astype(np.float32),
             'input_jet':jet})

        tf_y = tf.data.Dataset.from_tensor_slices(np.stack([self.y,self.w],-1))

        del self.X, self.y,  self.mask
        gc.collect()
        
        return tf.data.Dataset.zip((tf_zip,tf_y)).cache().shuffle(self.batch_size*100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)



class H1DataLoader(DataLoader):    
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.mean_part = [0.031, 0.0, -0.10,
                          -0.23,-0.10,0.27, 0.0,
                          0.0, 0.0,  0.0,  0.0,  0.0, 0.0]
        self.std_part = [0.35, 0.35,  0.178, 
                         1.2212526, 0.169,1.17,1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.mean_jet =  [ 19.15986358 , 0.57154217 , 6.00354102, 11.730992]
        self.std_jet  = [9.18613789, 0.80465287 ,2.99805704 ,5.14910232]
        
        self.load_data(path, batch_size,rank,size)
                
        self.y = np.identity(2)[self.y.astype(np.int32)]        
        self.num_pad = 5
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs        
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]
        

class OmniDataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.mean_jet =  [2.25826286e+02, 1.25739745e-03, 1.83963520e+01 ,1.88828832e+01]
        self.std_jet  = [90.39824296 , 1.34598289 ,10.73467645  ,8.45697634]

        
        self.path = path
        self.X = h5.File(self.path,'r')['reco'][rank::size]
        self.Y = h5.File(self.path,'r')['gen'][rank::size]        

        self.weight = np.ones(self.X.shape[0])
        
        self.nevts = h5.File(self.path,'r')['reco'].shape[0]
        self.num_part = self.X.shape[1]
        self.num_pad = 0

        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        self.num_jet = 4
        self.num_classes = 1
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

        self.reco = self.get_inputs(self.X,h5.File(self.path,'r')['reco_jets'][rank::size])
        self.gen = self.get_inputs(self.Y,h5.File(self.path,'r')['gen_jets'][rank::size])
        self.high_level_reco = h5.File(self.path,'r')['reco_subs'][rank::size]
        self.high_level_gen = h5.File(self.path,'r')['gen_subs'][rank::size]

    def get_inputs(self,X,jet):
        mask = X[:,:,2]!=0
        
        time = np.zeros((mask.shape[0],1)) #classifier gets time always 0
        #Preprocess and pad
        X = self.preprocess(X,mask).astype(np.float32)
        X = self.pad(X,num_pad=self.num_pad)
        jet = self.preprocess_jet(jet).astype(np.float32)
        coord = X[:,:,:2]
        return [X,coord,mask,jet,time]


    def data_from_file(self,file_path):
        with h5.File(file_path, 'r') as file:
            X = h5.File(file_path,'r')['reco'][:]
            reco = self.get_inputs(X,h5.File(file_path,'r')['reco_jets'][:])
            label_chunk = np.ones(X.shape[0])
                        
        return reco,label_chunk


class QGDataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size)        
        self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs        
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]


class CMSQGDataLoader(DataLoader):
    def __init__(self, path, batch_size=512,rank=0,size=1):
        super().__init__(path, batch_size, rank, size)

        self.load_data(path, batch_size,rank,size)
        self.y = np.identity(2)[self.y.astype(np.int32)]
        self.num_pad = 0
        self.num_feat = self.X.shape[2] + self.num_pad #missing inputs
        self.num_classes = self.y.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]

        
    
class JetClassDataLoader(DataLoader):
    def __init__(self, path,
                 batch_size=512,rank=0,size=1,chunk_size=5000, **kwargs):
        super().__init__(path, batch_size, rank, size)
        self.chunk_size = chunk_size

        all_files = [os.path.join(self.path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        self.files = np.array_split(all_files,self.size)[self.rank]

        self.get_stats(all_files)
        

    def get_stats(self,file_list):
        #Will assume each file is 100k long
        self.nevts = len(file_list)*100000//5
        self.num_part = h5.File(file_list[0],'r')['data'].shape[1]
        self.num_feat = h5.File(file_list[0],'r')['data'].shape[2]
        self.num_jet = 4 #hardcoded for convenience
        self.num_classes = h5.File(file_list[0],'r')['pid'].shape[1]
        self.steps_per_epoch = self.nevts//self.size//self.batch_size
        self.num_pad = 0
        
    def single_file_generator(self, file_path):
        with h5.File(file_path, 'r') as file:
            data_size = file['data'].shape[0]
            for start in range(0, data_size, self.chunk_size):
                end = min(start + self.chunk_size, data_size)
                jet_chunk = file['jet'][start:end]
                mask_particle = jet_chunk[:,-1] > 1
                jet_chunk = jet_chunk[mask_particle]
                data_chunk = file['data'][start:end].astype(np.float32)
                data_chunk = data_chunk[mask_particle]
                mask_chunk = data_chunk[:, :, 2] != 0  
                
                
                label_chunk = file['pid'][start:end]
                label_chunk = label_chunk[mask_particle]
                data_chunk = self.preprocess(data_chunk, mask_chunk).astype(np.float32)
                jet_chunk = self.preprocess_jet(jet_chunk).astype(np.float32)
                points_chunk = data_chunk[:, :, :2]
                for j in range(data_chunk.shape[0]):                        
                    yield ({
                        'input_features': data_chunk[j],
                        'input_points': points_chunk[j],
                        'input_mask': mask_chunk[j],
                        'input_jet':jet_chunk[j]},                           
                           label_chunk[j])
                    
                
    def interleaved_file_generator(self):
        random.shuffle(self.files)
        generators = [self.single_file_generator(fp) for fp in self.files]
        round_robin_generators = itertools.cycle(generators)

        while True:
            try:
                next_gen = next(round_robin_generators)
                yield next(next_gen)
            except StopIteration:
                break

    def make_tfdata(self):
        dataset = tf.data.Dataset.from_generator(
            self.interleaved_file_generator,
            output_signature=(
                {'input_features': tf.TensorSpec(shape=(self.num_part, self.num_feat), dtype=tf.float32),
                 'input_points': tf.TensorSpec(shape=(self.num_part, 2), dtype=tf.float32),
                 'input_mask': tf.TensorSpec(shape=(self.num_part), dtype=tf.float32),
                 'input_jet': tf.TensorSpec(shape=(self.num_jet), dtype=tf.float32)},
                tf.TensorSpec(shape=(self.num_classes), dtype=tf.int64)
            ))
        return dataset.shuffle(self.batch_size*50).repeat().batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
