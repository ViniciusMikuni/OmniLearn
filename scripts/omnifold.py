import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import os, gc
import horovod.tensorflow.keras as hvd
from tensorflow.keras.losses import mse
from scipy.special import expit
from PET import PET
import pickle

def weighted_binary_crossentropy(y_true, y_pred):
    """Custom loss function with weighted binary cross-entropy."""
    weights = tf.cast(tf.gather(y_true, [1], axis=1), tf.float32)  # Event weights
    y_true = tf.cast(tf.gather(y_true, [0], axis=1), tf.float32)  # Actual labels

    # Compute loss using TensorFlow's built-in function to handle numerical stability
    loss = weights * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(loss)


def convert_to_dict(x):    
    keys = ['input_features','input_points','input_mask','input_jet','input_time']
    d = {}
    for ik, key in enumerate(keys):
        d[key] = x[ik]

    return d
def concat_data(list1,list2):
    concatenated_arrays = []
    for arr1, arr2 in zip(list1, list2):
        concatenated_array = np.concatenate((arr1, arr2), axis=0)
        concatenated_arrays.append(concatenated_array)
    return concatenated_arrays


class OmniFold():
    """Main class for the OmniFold algorithm."""
    def __init__(self,version,num_iter,checkpoint_folder,
                 batch_size=512,epochs=200,size=1,
                 wd = 0.1,b1=0.95,b2=0.99,learning_rate_factor = 5.0,
                 learning_rate=1e-4,fine_tune=False):
        
        self.version = version
        self.num_iter = num_iter
        self.mc = None
        self.data=None
        self.batch_size=batch_size
        self.epochs=epochs
        self.learning_rate = learning_rate
        self.size = size
        self.fine_tune = fine_tune
        self.wd = wd
        self.b1 = b1
        self.b2 = b2
        if self.fine_tune:
            self.learning_rate_factor = learning_rate_factor
        else:
            self.learning_rate_factor = 1.

        self.checkpoint_folder = checkpoint_folder
            
    def Unfold(self):
                                        
        self.weights_pull = np.ones(self.mc.weight.shape[0])
        self.weights_push = np.ones(self.mc.weight.shape[0])
        self.CompileModel(self.learning_rate)
        for i in range(self.num_iter):
            if hvd.rank()==0:print("ITERATION: {}".format(i + 1))
            self.RunStep1(i)
            self.RunStep2(i)
            self.CompileModel(self.learning_rate,fixed=True)
            

    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        if hvd.rank()==0:print("RUNNING STEP 1")
        
        self.RunModel(
            concat_data(self.mc.reco, self.data.reco),
            np.concatenate((self.labels_mc, self.labels_data)),
            np.concatenate((self.weights_push*self.mc.weight,self.data.weight)),
            i,self.model1,stepn=1,
        )

        new_weights = self.reweight(self.mc.reco,self.model1)
        self.weights_pull = self.weights_push *new_weights
        #Ensure new set of weights dont modify the overall normalization
        norm_factor = hvd.allreduce(tf.constant(self.weights_pull,dtype=tf.float32)).numpy()
        self.weights_pull /= norm_factor

    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        if hvd.rank()==0:print("RUNNING STEP 2")
        
        self.RunModel(
            concat_data(self.mc.gen, self.mc.gen),
            np.concatenate((self.labels_mc, self.labels_gen)),
            np.concatenate((self.mc.weight, self.mc.weight*self.weights_pull)),
            i,self.model2,stepn=2,
        )
        new_weights=self.reweight(self.mc.gen,self.model2)
        norm_factor = hvd.allreduce(tf.constant(new_weights,dtype=tf.float32)).numpy()
        self.weights_push = new_weights/norm_factor

    def RunModel(self,
                 data,
                 labels,
                 weights,
                 iteration,
                 model,
                 stepn,
                 cached = False,
                 ):

        
        verbose = 1 if hvd.rank() == 0 else 0        
        permutation = np.random.permutation(labels.shape[0])

        #Shuffle
        data = [arr[permutation] for arr in data]
        data = convert_to_dict(data)
        labels = labels[permutation]
        weights = weights[permutation]
        y = np.stack((labels,weights),axis=1)

        
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            
            ReduceLROnPlateau(patience=100, min_lr=1e-6, monitor="val_loss"),
            EarlyStopping(patience=3,restore_best_weights=True,monitor="val_loss"),
        ]
        
        
        if hvd.rank() ==0:
        
            model_name = '{}/checkpoints/OmniFold_{}_iter{}_step{}.weights.h5'.format(
                self.checkpoint_folder,self.version,iteration,stepn)
            callbacks.append(ModelCheckpoint(model_name,save_best_only=True,
                                             mode='auto',period=1,save_weights_only=True))
                    
        hist =  model.fit(
            data,y,
            batch_size = self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
            shuffle=True,
            #validation_data = (X_train,y_train),
            verbose = verbose,
            callbacks=callbacks)
        
        if hvd.rank() ==0:
            with open(model_name.replace("checkpoints","histories").replace(".weights.h5",".pkl"),"wb") as f:
                pickle.dump(hist.history, f)

                
    def Preprocessing(self,model1,model2):
        self.PrepareInputs()
        self.PrepareModel(model1,model2)


    def CompileModel(self,learning_rate,fixed=False):
        
        learning_rate_schedule_body = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate/self.learning_rate_factor,
            warmup_target = learning_rate*np.sqrt(self.size)/self.learning_rate_factor,
            warmup_steps= 3*(self.mc.nevts + self.data.nevts)//self.batch_size//self.size,
            decay_steps= self.epochs*(self.mc.nevts + self.data.nevts)//self.batch_size//self.size,
            alpha = 1e-2,
        )


        learning_rate_schedule_head = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            warmup_target = learning_rate*np.sqrt(self.size),
            warmup_steps= 3*(self.mc.nevts + self.data.nevts)//self.batch_size//self.size,
            decay_steps= self.epochs*(self.mc.nevts + self.data.nevts)//self.batch_size//self.size,
            alpha = 1e-2,
        )

        min_learning_rate = 1e-6
        opt_head1 = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else learning_rate_schedule_head,
            weight_decay=self.wd,
            beta_1=self.b1,
            beta_2=self.b2)
        
        opt_head1 = hvd.DistributedOptimizer(opt_head1)
        
        opt_body1 = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else learning_rate_schedule_body,
            weight_decay=self.wd,
            beta_1=self.b1,
            beta_2=self.b2)
        
        opt_body1 = hvd.DistributedOptimizer(opt_body1)


        opt_head2 = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else learning_rate_schedule_head,
            weight_decay=self.wd,
            beta_1=self.b1,
            beta_2=self.b2)
        
        opt_head2 = hvd.DistributedOptimizer(opt_head2)
        
        opt_body2 = tf.keras.optimizers.Lion(
            learning_rate=min_learning_rate if fixed else learning_rate_schedule_body,
            weight_decay=self.wd,
            beta_1=self.b1,
            beta_2=self.b2)
        
        opt_body2 = hvd.DistributedOptimizer(opt_body2)


        self.model1.compile(opt_body1,opt_head1)
        self.model2.compile(opt_body2,opt_head2)

    def PrepareInputs(self):
        self.labels_mc = np.zeros(self.mc.weight.shape[0],dtype=np.float32)
        self.labels_data = np.ones(self.data.weight.shape[0],dtype=np.float32)
        self.labels_gen = np.ones(self.mc.weight.shape[0],dtype=np.float32)


    def PrepareModel(self,model1,model2):
        self.model1 = model1
        self.model2 = model2

    def reweight(self,events,model,batch_size=None):
        if batch_size is None:
           batch_size =  self.batch_size
        f = np.nan_to_num(expit(model.predict(events,batch_size=batch_size,verbose=0)[0])
                          ,posinf=1,neginf=0)
        weights = f / (1.0 -f)
        #weights = weights[:,0]
        return np.squeeze(np.nan_to_num(weights,posinf=1))

class Classifier(keras.Model):
    def __init__(self,
                 num_feat,
                 num_jet,      
                 num_classes=2,
                 num_drop = 7, 
                 feature_drop = 0.2,
                 projection_dim = 128,
                 local = True, K = 10,
                 num_local = 2, 
                 num_layers = 8, num_class_layers=2,
                 num_heads = 4,drop_probability = 0.0,
                 simple = False, layer_scale = True,
                 layer_scale_init = 1e-5,        
                 talking_head = False,
                 mode = 'classifier',
                 class_activation='sigmoid',
                 fine_tune = False,
                 model_name = None,
                 ):
        super(Classifier, self).__init__()
        model = PET(num_feat=num_feat,
                    num_jet=num_jet,
                    num_classes=num_classes,
                    local = local,
                    num_layers = num_layers, 
                    drop_probability = drop_probability,
                    #dropout=0.1,
                    simple = simple, layer_scale = layer_scale,
                    talking_head = talking_head,
                    mode = mode,
                    class_activation=class_activation,
                    )
        if fine_tune:
            assert model_name is not None, "ERROR: Model name is necessary if fine tune is on"
            model.load_weights(model_name,by_name=True,skip_mismatch=True)
            #self.model.body.trainable=False
        
        self.head = model.classifier_head
        self.body = model.body
        self.model = model.classifier
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self,x):
        return self.model(x)

    def compile(self,body_optimizer,head_optimizer):
        super(Classifier, self).compile(experimental_run_tf_function=False,
                                        weighted_metrics=[],
                                        #run_eagerly=True
        )
        self.body_optimizer = body_optimizer
        self.optimizer = head_optimizer


    def train_step(self, inputs):
        x,y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        x['input_time'] = tf.zeros((batch_size,1))
        with tf.GradientTape(persistent=True) as tape:
            y_pred,y_evt = self.model(x)
            loss_pred = weighted_binary_crossentropy(y, y_pred)
            loss_evt = mse(x['input_jet'],y_evt)
            loss = loss_pred+loss_evt


        self.body_optimizer.minimize(loss_pred,self.body.trainable_variables,tape=tape)
        self.optimizer.minimize(loss,self.head.trainable_variables,tape=tape)

        
        self.loss_tracker.update_state(loss_pred)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, inputs):
        x,y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        x['input_time'] = tf.zeros((batch_size,1))

        y_pred,y_evt = self.model(x)
        loss_evt = mse(x['input_jet'],y_evt)
        loss_pred = weighted_binary_crossentropy(y, y_pred)
        loss = loss_pred+loss_evt
        self.loss_tracker.update_state(loss_pred)
        return {"loss": self.loss_tracker.result()}

