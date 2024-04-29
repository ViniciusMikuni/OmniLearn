import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import utils
from tensorflow.keras.losses import mse, mae
from tensorflow.keras.models import Model
from PET import PET, FourierProjection, get_encoding
#import horovod.tensorflow as hvd
from layers import StochasticDepth,LayerScale
from tqdm import tqdm

class PET_jetnet(keras.Model):
    """Score based generative model"""
    def __init__(self,
                 num_feat,
                 num_jet,      
                 num_classes=2,
                 num_part = 150,
                 num_diffusion = 3,
                 feature_drop = 0.1,
                 projection_dim = 128,
                 local = True, K = 10,
                 num_local = 2, 
                 num_layers = 8, num_class_layers=2,
                 num_heads = 4,drop_probability = 0.0,
                 simple = False, layer_scale = True,
                 layer_scale_init = 1e-5,        
                 talking_head = False,
                 mode = 'generator',                 
                 fine_tune = False,
                 model_name = None,
                 use_mean=False):
        super(PET_jetnet, self).__init__()


        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_classes = num_classes
        self.max_part = num_part
        self.projection_dim = projection_dim
        self.layer_scale_init = layer_scale_init
        self.num_steps = 300
        self.num_diffusion = num_diffusion
        self.ema=0.999
        self.shape = (-1,1,1)

        self.model_part  = PET(num_feat=num_feat,
                               num_jet=num_jet,
                               num_classes=num_classes,
                               local = local,
                               num_layers = num_layers, 
                               drop_probability = drop_probability,
                               simple = simple, layer_scale = layer_scale,
                               talking_head = talking_head,
                               mode = mode,                               
                               )


        if fine_tune:
            assert model_name is not None, "ERROR: Model name is necessary if fine tune is on"
            self.model_part.load_weights(model_name,by_name=True,skip_mismatch=True)
            #self.model_part.ema_body.trainable=False            

            
        if use_mean:
            self.mean, self.std = self.get_mean()
        else:
            self.mean = 0.0
            self.std = 1.0            

        self.body = self.model_part.ema_body        
        self.head = self.model_part.ema_generator_head
                
                
        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self.num_classes))
        inputs_jet = Input((self.num_jet))
        inputs_mask = Input((None,1))
        inputs_features = Input(shape=(None, num_feat))
        inputs_points = Input(shape=(None, 2))


        x = inputs_mask*(inputs_features-self.mean)/self.std
        
        output_body = self.body([x,inputs_points,inputs_mask,inputs_time])
        outputs_head = self.head([output_body,inputs_jet,inputs_mask,inputs_time,inputs_cond])
        outputs = inputs_mask*(self.std*outputs_head + self.mean)
        
        self.model_part = keras.Model(inputs=[inputs_features,inputs_points,inputs_mask,
                                              inputs_jet,inputs_time,inputs_cond],
                                      outputs=outputs)
        

                        
        outputs = self.Resnet(
            inputs_jet,
            inputs_time,
            inputs_cond,
            num_layer = 3,
            mlp_dim= 2*self.projection_dim,
        )

        

        self.model_jet = Model(inputs=[inputs_jet,inputs_time,inputs_cond],
                                     outputs=outputs)

            
        self.ema_jet = keras.models.clone_model(self.model_jet)
        self.ema_body = keras.models.clone_model(self.body)
        self.ema_head = keras.models.clone_model(self.head)

        #self.ema_part = keras.models.clone_model(self.model_part)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_part_tracker = keras.metrics.Mean(name="part")
        self.loss_jet_tracker = keras.metrics.Mean(name="jet")

        self.multistep_coefficients = [
            tf.constant([1], shape=(1, 1, 1, 1), dtype=tf.float32),
            tf.constant([-1, 3], shape=(2, 1, 1, 1), dtype=tf.float32) / 2,
            tf.constant([5, -16, 23], shape=(3, 1, 1,1), dtype=tf.float32) / 12,
            tf.constant([-9, 37, -59, 55], shape=(4, 1,1, 1), dtype=tf.float32)
            / 24,
            tf.constant(
                [251, -1274, 2616, -2774, 1901], shape=(5, 1,1, 1), dtype=tf.float32
            )
            / 720,
        ]
        

    def get_mean(self):
        #Mean and std from JetClass pretrained model to be used during fine-tuning
        mean_pet = tf.constant([0.0, 0.0,-0.0278,
                                0.0,0.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,0.0],
                               shape=(1, 1, self.num_feat), dtype=tf.float32)
        std_pet = tf.constant([0.215,0.215,0.070,
                               1.0,1.0,1.0,1.0,1.0,
                               1.0,1.0,1.0,1.0,1.0],
                              shape=(1, 1, self.num_feat), dtype=tf.float32)

        if self.max_part == 150:
            mean_sample = tf.constant([0.0, 0.0, -0.0217,
                                       0.0,0.0,0.0,0.0,0.0,
                                       0.0,0.0,0.0,0.0,0.0],
                                      shape=(1, 1, self.num_feat), dtype=tf.float32)
            std_sample =  tf.constant([0.115, 0.115, -0.054,
                                       1.0,1.0,1.0,1.0,1.0,
                                       1.0,1.0,1.0,1.0,1.0],
                                      shape=(1, 1, self.num_feat), dtype=tf.float32)
        elif self.max_part == 30:
            mean_sample = tf.constant([0.0, 0.0, -0.035,
                                       0.0,0.0,0.0,0.0,0.0,
                                       0.0,0.0,0.0,0.0,0.0],
                                      shape=(1, 1, self.num_feat), dtype=tf.float32)
            std_sample = tf.constant([0.09, 0.09,  0.067, 
                                      1.0,1.0,1.0,1.0,1.0,
                                      1.0,1.0,1.0,1.0,1.0],
                                     shape=(1, 1, self.num_feat), dtype=tf.float32)

        return (mean_sample-mean_pet)/std_pet, std_sample/std_pet

        
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker,self.loss_part_tracker,self.loss_jet_tracker]


    def compile(self,body_optimizer,head_optimizer):
        super(PET_jetnet, self).compile(experimental_run_tf_function=False,
                                        weighted_metrics=[],
                                        #run_eagerly=True
        )
        self.body_optimizer = body_optimizer
        self.optimizer = head_optimizer

    

    
    def Resnet(self,
               inputs,
               inputs_time,
               labels,               
               num_layer = 3,
               mlp_dim=128,
               dropout=0.0,
               ):
    
        def resnet_dense(input_layer,hidden_size,nlayers=2):
            x = input_layer
            residual = layers.Dense(hidden_size)(x)
            for _ in range(nlayers):
                x = layers.Dense(hidden_size,activation='swish')(x)
                x = layers.Dropout(dropout)(x)
            x = LayerScale(self.layer_scale_init,hidden_size)(x)
            return residual + x

        time = FourierProjection(inputs_time,self.projection_dim)
        cond_token = layers.Dense(self.projection_dim)(labels)
        cond_token = layers.Dense(2*self.projection_dim,activation='gelu')(cond_token + time)
        scale,shift = tf.split(cond_token,2,-1)
        
        layer = layers.Dense(self.projection_dim,activation='swish')(inputs)
        #layer = layers.LayerNormalization(epsilon=1e-6)(layer)
        layer = layer*(1.0+scale) + shift
        
        for _ in range(num_layer-1):
            layer = layers.LayerNormalization(epsilon=1e-6)(layer)
            layer =  resnet_dense(layer,mlp_dim)

        layer = layers.LayerNormalization(epsilon=1e-6)(layer)
        outputs = layers.Dense(self.num_jet,kernel_initializer="zeros")(layer)
    
        return outputs


    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)
    

    def train_step(self, inputs):
        x,y = inputs
        batch_size = tf.shape(x['input_jet'])[0]

        with tf.GradientTape(persistent=True) as tape:            
            t = tf.random.uniform((batch_size,1))                
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)
            
            eps = tf.random.normal((tf.shape(x['input_features'])),
                                   dtype=tf.float32)*x['input_mask'][:,:,None]

            mask_diffusion = tf.concat([
                tf.ones_like(eps[:, :, :self.num_diffusion], dtype=tf.bool),
                tf.zeros_like(eps[:, :, self.num_diffusion:], dtype=tf.bool)
            ], axis=-1)

            
            #zero entries not needed
            eps = tf.where(mask_diffusion, eps, tf.zeros_like(eps))
                
            perturbed_x = alpha[:,None]*x['input_features'] + eps * sigma[:,None]
            perturbed_x = tf.where(mask_diffusion, perturbed_x, tf.zeros_like(perturbed_x))
                        
            v_pred_part = self.model_part([perturbed_x,
                                           perturbed_x[:,:,:2],
                                           x['input_mask'],
                                           x['input_jet'],t,y])
            v_pred_part = tf.reshape(v_pred_part[:,:,:self.num_diffusion],(tf.shape(v_pred_part)[0], -1))
            v_part = alpha[:,None] * eps - sigma[:,None] * x['input_features']
            v_part = tf.reshape(v_part[:,:,:self.num_diffusion],(tf.shape(v_part)[0], -1))


            loss_part = tf.reduce_sum(tf.square(v_part-v_pred_part))/(self.num_diffusion*tf.reduce_sum(x['input_mask']))
            #loss_part = mse(v_part,v_pred_part)

        
            #Jet model

            eps = tf.random.normal((batch_size,self.num_jet),dtype=tf.float32)
            perturbed_x = alpha*x['input_jet'] + eps * sigma            
            v_pred = self.model_jet([perturbed_x,t,y])
            
            v_jet = alpha * eps - sigma * x['input_jet']
            loss_jet = tf.reduce_mean(tf.square(v_pred-v_jet))

            loss = loss_jet + loss_part



        self.body_optimizer.minimize(loss_part,self.body.trainable_variables,tape=tape)
                   
        trainable_vars = self.model_jet.trainable_variables + self.head.trainable_variables
        #+self.scaler_input.trainable_variables
        self.optimizer.minimize(loss,trainable_vars,tape=tape)

        

        self.loss_tracker.update_state(loss)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)
            
        for weight, ema_weight in zip(self.model_jet.weights, self.ema_jet.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
            
        for weight, ema_weight in zip(self.head.weights, self.ema_head.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        for weight, ema_weight in zip(self.body.weights, self.ema_body.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)


        return {m.name: m.result() for m in self.metrics}


    def test_step(self, inputs):
        x,y = inputs
        batch_size = tf.shape(x['input_jet'])[0]

        t = tf.random.uniform((batch_size,1))                
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)

        eps = tf.random.normal((tf.shape(x['input_features'])),
                               dtype=tf.float32)*x['input_mask'][:,:,None]
        
        mask_diffusion = tf.concat([
            tf.ones_like(eps[:, :, :self.num_diffusion], dtype=tf.bool),
            tf.zeros_like(eps[:, :, self.num_diffusion:], dtype=tf.bool)
        ], axis=-1)
        
            
        #zero entries not needed
        eps = tf.where(mask_diffusion, eps, tf.zeros_like(eps))
        
        perturbed_x = alpha[:,None]*x['input_features'] + eps * sigma[:,None]
        perturbed_x = tf.where(mask_diffusion, perturbed_x, tf.zeros_like(perturbed_x))
        
        v_pred_part = self.model_part([perturbed_x,
                                       perturbed_x[:,:,:2],
                                       x['input_mask'],
                                       x['input_jet'],t,y])
        v_pred_part = tf.reshape(v_pred_part[:,:,:self.num_diffusion],(tf.shape(v_pred_part)[0], -1))
        v_part = alpha[:,None] * eps - sigma[:,None] * x['input_features']
        v_part = tf.reshape(v_part[:,:,:self.num_diffusion],(tf.shape(v_part)[0], -1))
        loss_part = tf.reduce_sum(tf.square(v_part-v_pred_part))/(self.num_diffusion*tf.reduce_sum(x['input_mask']))
        #loss_part = mse(v_part,v_pred_part)
                
            
        #Jet model

        eps = tf.random.normal((batch_size,self.num_jet),dtype=tf.float32)
        perturbed_x = alpha*x['input_jet'] + eps * sigma        
        v_pred = self.model_jet([perturbed_x,t,y])

        v_jet = alpha * eps - sigma * x['input_jet']
        loss_jet = tf.reduce_mean(tf.square(v_pred-v_jet))
            
        self.loss_tracker.update_state(loss_part)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)
            
        return {m.name: m.result() for m in self.metrics}
            
    def call(self,x):        
        return self.model(x)

    def generate(self,cond,nsplit = 2,jets=None,use_tqdm=False):
        jet_info = []
        part_info = []

        if jets is not None:
            jet_split = np.array_split(jets,nsplit)
            
        splits = np.array_split(cond, nsplit)
        
        #iterable = tqdm(splits,desc='Processing Splits',total=len(splits)) if use_tqdm else splits
        for i, split in tqdm(enumerate(splits), total=len(splits), desc='Processing Splits') if use_tqdm else enumerate(splits):
            if jets is not None:
                jet = jet_split[i]
            else:
                
                jet = self.DDPMSampler(split,self.ema_jet,
                                       data_shape=[split.shape[0],self.num_jet],
                                       w = 0.0,
                                       num_steps = 512,
                                       const_shape = [-1,1]).numpy()

            jet_info.append(jet)
            
            nparts = np.expand_dims(np.clip(utils.revert_npart(jet[:,-1],name=str(self.max_part)),
                                            5,self.max_part),-1) #5 is the minimum in the datasets used for training

            mask = np.expand_dims(
                np.tile(np.arange(self.max_part),(nparts.shape[0],1)) < np.tile(nparts,(1,self.max_part)),-1)
        
            assert np.sum(np.sum(mask.reshape(mask.shape[0],-1),-1,keepdims=True)-nparts)==0, 'ERROR: Particle mask does not match the expected number of particles'

            parts = self.DDPMSampler(split,[self.ema_body,self.ema_head],
                                     data_shape=[split.shape[0],self.max_part,self.num_feat],
                                     jet=jet,
                                     num_steps = self.num_steps,
                                     const_shape = self.shape,
                                     w = 0.0,
                                     mask=mask.astype(np.float32)).numpy()
            part_info.append(parts*mask)            
        return np.concatenate(part_info),np.concatenate(jet_info)
    

    def logsnr_schedule_cosine(self,t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))

    def inv_logsnr_schedule_cosine(self,logsnr, logsnr_min=-10., logsnr_max=20.):
        #if self.ll_training:return (logsnr_max - logsnr)/(logsnr_max - logsnr_min)
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return tf.math.atan(tf.exp(-0.5 * tf.cast(logsnr,tf.float32)))/a -b/a


    def get_logsnr_alpha_sigma(self,time,shape=None):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))

        if shape is not None:
            alpha = tf.reshape(alpha, shape)
            sigma = tf.reshape(sigma, shape)
            logsnr = tf.reshape(logsnr,shape)
            
        return logsnr, tf.cast(alpha,tf.float32), tf.cast(sigma,tf.float32)


    @tf.function
    def NoisySampler(self,
                     cond,
                     model,
                     data_shape=None,
                     const_shape=None,
                     jet=None,
                     w = 0.1,
                     num_steps = 100,
                     mask=None):

        batch_size = cond.shape[0]
        x = self.prior_sde(data_shape)
        if jet is not None:
            mask_diffusion = tf.concat([
                tf.ones_like(x[:, :, :self.num_diffusion], dtype=tf.bool),
                tf.zeros_like(x[:, :, self.num_diffusion:], dtype=tf.bool)
            ], axis=-1)

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            logsnr_t, alpha, sigma = self.get_logsnr_alpha_sigma(t,shape=const_shape)
            logsnr_s, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / self.num_steps,shape=const_shape)

            if jet is None:
                v = model([x, t, cond], training=False)
                #- w*model([x, sigma**2, tf.zeros_like(cond)], training=False)
            else:
                x = tf.where(mask_diffusion, x*mask, tf.zeros_like(x))                
                model_body, model_head = model
                v = self.evaluate_models(model_head,model_body,x,jet,mask,t,cond,w)
            
            pred_x = alpha * x - sigma * v
            
            alpha_st = tf.math.sqrt((1. + tf.exp(-logsnr_t)) / (1. + tf.exp(-logsnr_s)))
            r = tf.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
            one_minus_r = -tf.math.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
            
            mean = r * alpha_st * x + one_minus_r * alpha_ * pred_x
                      
            std = tf.sqrt(one_minus_r) * sigma
            eps = tf.random.normal(data_shape,dtype=tf.float32)
            x = mean + std*eps
            
        return pred_x

    

    @tf.function
    def second_order_correction(self,time_step,x,pred_images,pred_noises,
                                alphas,sigmas,w,
                                cond,model,jet=None,masks=None,
                                num_steps=100,
                                second_order_alpha=0.5,shape=None):
        step_size = 1.0/num_steps
        t = time_step - second_order_alpha * step_size
        logsnr, alpha_signal_rates, alpha_noise_rates = self.get_logsnr_alpha_sigma(t,shape=shape)
        alpha_noisy_images = alpha_signal_rates * pred_images + alpha_noise_rates * pred_noises

        if jet is None:
            v = model([alpha_noisy_images, t,cond],training=False)
            #- w*model([alpha_noisy_images,t,tf.zeros_like(cond)],training=False)
        else:
            mask,mask_diffusion = masks
            alpha_noisy_images = tf.where(mask_diffusion, alpha_noisy_images*mask, tf.zeros_like(x))
            model_body, model_head = model
            v = self.evaluate_models(model_head,model_body,
                                     alpha_noisy_images,
                                     jet,mask,t,cond,w)
            
        alpha_pred_noises = alpha_noise_rates * alpha_noisy_images + alpha_signal_rates * v
        # linearly combine the two noise estimates
        pred_noises = (1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises + 1.0 / (
            2.0 * second_order_alpha
        ) * alpha_pred_noises

        mean = (x - sigmas * pred_noises) / alphas        
        eps = pred_noises
        
        return mean,eps
    
    def evaluate_models(self,head,body,x,jet,mask,t,cond,w = 0.0):
        x_in = mask*(x-self.mean)/self.std
        v = body([x_in,x[:,:,:2],mask,t], training=False)
        v = (1.0+w)*head([v,jet,mask,t,cond],training=False)  - w*head([v,jet,mask,t,tf.zeros_like(cond)],training=False)
        return mask*(self.std*v + self.mean)


           
    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    jet=None,
                    w = 0.1,
                    num_steps = 100,
                    mask=None):
        """Generate samples from score-based models with DDPM method.
        
        Args:
        cond: Conditional input
        model: Trained score model to use
        data_shape: Format of the data
        const_shape: Format for constants, should match the data_shape in dimensions
        jet: input jet conditional information if used
        mask: particle mask if used

        Returns: 
        Samples.
        """

        batch_size = cond.shape[0]
        x = self.prior_sde(data_shape)
        if jet is not None:
            mask_diffusion = tf.concat([
                tf.ones_like(x[:, :, :self.num_diffusion], dtype=tf.bool),
                tf.zeros_like(x[:, :, self.num_diffusion:], dtype=tf.bool)
            ], axis=-1)

            
        prev_pred_noises = []  # only required for multistep sampling
        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t,shape=const_shape)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps,shape=const_shape)
            if jet is None:
                v = model([x, t, cond], training=False)                
                masks = None
            else:
                x = tf.where(mask_diffusion, x*mask, tf.zeros_like(x))
                model_body, model_head = model
                v = self.evaluate_models(model_head,model_body,x,jet,mask,t,cond,w)
                masks = [mask,mask_diffusion]
                                                                                
            mean = alpha * x - sigma * v
            eps = v * alpha + x * sigma
            mean,eps = self.second_order_correction(t,x,mean,eps,
                                                    alpha,sigma,w,
                                                    cond,model,jet,masks,
                                                    num_steps=num_steps,
                                                    shape=const_shape
                                                    )

            x = alpha_ * mean + sigma_ * eps
        return mean

    def multistep_correction(
        self, noisy_images, signal_rates, noise_rates, prev_pred_noises, num_multisteps
    ):
        # Adams-Bashforth multistep method
        # https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Bashforth_methods
        # based on https://arxiv.org/abs/2202.09778

        # linearly combine previous noise estimates
        # doing this with the image component leads to identical results
        pred_noises = tf.reduce_sum(
            self.multistep_coefficients[len(prev_pred_noises) - 1]
            * tf.stack(prev_pred_noises, axis=0),
            axis=0,
        )
        if len(prev_pred_noises) == num_multisteps:
            # remove oldest noise estimate
            prev_pred_noises.pop(0)

        # recalculate component estimates

        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates        
        return pred_images, pred_noises
