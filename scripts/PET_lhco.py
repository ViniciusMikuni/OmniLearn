import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import utils
from tensorflow.keras.losses import mse, binary_crossentropy,mae
from tensorflow.keras.models import Model
from PET import PET, FourierProjection, get_encoding
from layers import LayerScale, StochasticDepth
from omnifold import weighted_binary_crossentropy

class PET_lhco(keras.Model):
    """Score based generative model"""
    def __init__(self,
                 num_feat,
                 num_jet,      
                 num_classes=2,
                 num_part = 150,
                 num_diffusion=3,
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
        super(PET_lhco, self).__init__()


        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.max_part = num_part
        self.num_diffusion=num_diffusion
        self.projection_dim = projection_dim
        self.feature_drop = feature_drop
        self.layer_scale = layer_scale
        self.layer_scale_init=layer_scale_init
        self.num_steps = 512
        self.ema=0.999
        self.shape = (-1,1,1)

        self.model_part =  PET(num_feat=num_feat,
                               num_jet=num_jet,
                               num_classes=num_classes,
                               feature_drop=feature_drop,
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


        if use_mean:
            self.mean, self.std = self.get_mean()
        else:
            self.mean = 0.0
            self.std = 1.0

        

        self.body = self.model_part.ema_body
        self.head = self.model_part.ema_generator_head
        
        
        
        inputs_time = Input((1))
        inputs_cond = Input((self.num_classes))
        inputs_jet = Input((None,self.num_jet))
        input_jet = Input((self.num_jet)) #for conditional information fed into the particle generation model
        inputs_mask = Input((None,1))
        inputs_features = Input(shape=(None, num_feat))
        inputs_points = Input(shape=(None, 2))


        x = inputs_mask*(inputs_features-self.mean)/self.std
        x = self.body([x,inputs_points,inputs_mask,inputs_time])
        outputs_head = self.head([x,input_jet,inputs_mask,inputs_time,inputs_cond])
        outputs = inputs_mask*(self.std*outputs_head + self.mean)

        # outputs = self.DeepSets(inputs_features,inputs_mask,input_jet,inputs_cond,inputs_time)
        
        #inputs_mask
        self.model_part = keras.Model(inputs=[inputs_features,inputs_points,inputs_mask,
                                          input_jet,inputs_time,inputs_cond],
                                      outputs=outputs)
        
                            
        outputs = self.small_PET(
            inputs_jet,
            inputs_cond,
            inputs_time,
        )


        self.model_jet = Model(inputs=[inputs_jet,inputs_time,inputs_cond],
                               outputs=outputs)

            
        self.ema_jet = keras.models.clone_model(self.model_jet)
        self.ema_body = keras.models.clone_model(self.body)
        self.ema_head = keras.models.clone_model(self.head)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_part_tracker = keras.metrics.Mean(name="part")
        self.loss_jet_tracker = keras.metrics.Mean(name="jet")

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

        mean_sample = tf.constant([0.0, 0.0, -0.019,
                                   0.0,0.0,0.0,0.0,0.0,
                                   0.0,0.0,0.0,0.0,0.0],
                                  shape=(1, 1, self.num_feat), dtype=tf.float32)
        
        std_sample = tf.constant([0.26,0.26,0.066, 
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
        super(PET_lhco, self).compile(experimental_run_tf_function=False,
                                        weighted_metrics=[],
                                        #run_eagerly=True
        )
        self.body_optimizer = body_optimizer
        self.optimizer = head_optimizer
      
    def small_PET(self,
                  input_features,
                  input_label,
                  time,
                  num_layers = 6,
                  num_heads = 4,
                  ):

        time = FourierProjection(time,self.projection_dim)
        cond_info = layers.Dense(2*self.projection_dim,activation='gelu',use_bias=False)(input_label)
        cond_info = layers.Dense(self.projection_dim,activation='gelu',use_bias=False)(cond_info)
        cond_info = StochasticDepth(self.feature_drop)(cond_info)
        
        cond_info = layers.Add()([cond_info,time])
        cond_info = tf.tile(cond_info[:,None, :], [1,tf.shape(input_features)[1], 1])
        cond_info = layers.Dense(2*self.projection_dim,activation='gelu')(cond_info)
        scale,shift = tf.split(cond_info,2,-1)
        
        encoded = get_encoding(input_features,self.projection_dim)
        encoded = layers.GroupNormalization(groups=1)(encoded)*(1.0 + scale) + shift 
                                                   
        for i in range(num_layers):
            x1 = layers.GroupNormalization(groups=1)(encoded)
            updates = layers.MultiHeadAttention(num_heads=num_heads,
                                                key_dim=self.projection_dim//num_heads)(x1,x1)
            updates = layers.GroupNormalization(groups=1)(updates)
            if self.layer_scale:
                updates = LayerScale(self.layer_scale_init, self.projection_dim)(updates)
            
            x2 = layers.Add()([updates,encoded])
            x3 = layers.GroupNormalization(groups=1)(x2)
            x3 = layers.Dense(2*self.projection_dim,activation="gelu")(x3)
            x3 = layers.Dense(self.projection_dim)(x3)            
            if self.layer_scale:
                x3 = LayerScale(self.layer_scale_init, self.projection_dim)(x3)
            encoded = layers.Add()([x3,x2])

            
        # cond_info = layers.Dense(2*self.projection_dim,activation='gelu')(cond_info)
        # scale,shift = tf.split(cond_info,2,-1)
        # encoded = layers.GroupNormalization(groups=1)(encoded)*(1.0+scale) + shift

        encoded = layers.GroupNormalization(groups=1)(encoded)
        outputs = layers.Dense(self.num_jet,kernel_initializer="zeros")(encoded)
        
        return outputs


    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)
    

    def train_step(self, inputs):
        x,y = inputs
        batch_size = tf.shape(x['input_jet'])[0]

        
        num_part = tf.shape(x['input_features'])[2]

        with tf.GradientTape(persistent=True) as tape:
            #Jets
            t = tf.random.uniform((batch_size,1))                
            _, alpha, sigma = self.get_logsnr_alpha_sigma(t)
            eps = tf.random.normal((tf.shape(x['input_jet'])),
                                   dtype=tf.float32)
                
            perturbed_x = alpha*x['input_jet'] + eps * sigma
            
            v_pred_jet = self.model_jet([perturbed_x,t,y])
            v_pred_jet = tf.reshape(v_pred_jet,(tf.shape(v_pred_jet)[0], -1))
            
            v_jet = alpha * eps - sigma * x['input_jet']
            v_jet = tf.reshape(v_jet,(tf.shape(v_jet)[0], -1))
            loss_jet = mse(v_jet,v_pred_jet)



            #Particles, split the jet into 2, having effectively 2* batch size
            
            x['input_features'] = tf.reshape(x['input_features'],(-1,num_part,self.num_feat))
            x['input_mask'] = tf.reshape(x['input_mask'],(-1,num_part,1))
            x['input_jet'] = tf.reshape(x['input_jet'],(-1,self.num_jet))
            y = tf.reshape(tf.tile(y,(1,2)),(-1,self.num_classes))
            
            eps = tf.random.normal((tf.shape(x['input_features'])),
                                   dtype=tf.float32)
            mask_diffusion = tf.concat([
                tf.ones_like(eps[:, :, :self.num_diffusion], dtype=tf.bool),
                tf.zeros_like(eps[:, :, self.num_diffusion:], dtype=tf.bool)
            ], axis=-1)
            #zero entries not needed
            eps = tf.where(mask_diffusion, eps*x['input_mask'], tf.zeros_like(eps))
            

            t = tf.random.uniform((2*batch_size,1))                
            _, alpha, sigma = self.get_logsnr_alpha_sigma(t)
                
            perturbed_x = alpha*x['input_features'] + eps * sigma
            perturbed_x = tf.where(mask_diffusion,
                                   perturbed_x*x['input_mask'],
                                   tf.zeros_like(perturbed_x))
            
            v_pred_part = self.model_part([perturbed_x,
                                           perturbed_x[:,:,:2],
                                           x['input_mask'],
                                           x['input_jet'],t,y])


            v_pred_part = tf.reshape(v_pred_part[:,:,:self.num_diffusion],(tf.shape(v_pred_part)[0], -1))
            v_part = alpha * eps - sigma * x['input_features']
            v_part = tf.reshape(v_part[:,:,:self.num_diffusion],(tf.shape(v_part)[0], -1))

            loss_part = tf.reduce_sum(tf.square(v_part-v_pred_part))/(self.num_diffusion*tf.reduce_sum(x['input_mask']))
            
            

            loss = tf.reduce_mean(loss_jet) + tf.reduce_mean(loss_part)

            
        self.body_optimizer.minimize(loss_part,self.body.trainable_variables,tape=tape)            
        trainable_vars =  self.head.trainable_variables + self.model_jet.trainable_variables
        #self.head.trainable_variables +
        self.optimizer.minimize(loss,trainable_vars,tape=tape)


        

        self.loss_tracker.update_state(loss_jet)
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
        num_part = tf.shape(x['input_features'])[2]
        
        #Jets
        t = tf.random.uniform((batch_size,1))                
        _, alpha, sigma = self.get_logsnr_alpha_sigma(t)
        eps = tf.random.normal((tf.shape(x['input_jet'])),
                               dtype=tf.float32)
                
        perturbed_x = alpha*x['input_jet'] + eps * sigma

            
        v_pred_jet = self.model_jet([perturbed_x,t,y])
        v_pred_jet = tf.reshape(v_pred_jet,(tf.shape(v_pred_jet)[0], -1))
        
        v_jet = alpha * eps - sigma * x['input_jet']
        v_jet = tf.reshape(v_jet,(tf.shape(v_jet)[0], -1))
        loss_jet = mse(v_jet,v_pred_jet)
        
        # #Particles, split the jet into 2, having effectively 2* batch size
        
        
        x['input_features'] = tf.reshape(x['input_features'],(-1,num_part,self.num_feat))
        x['input_mask'] = tf.reshape(x['input_mask'],(-1,num_part,1))
        x['input_jet'] = tf.reshape(x['input_jet'],(-1,self.num_jet))
        y = tf.reshape(tf.tile(y,(1,2)),(-1,self.num_classes))
        
        eps = tf.random.normal((tf.shape(x['input_features'])),
                               dtype=tf.float32)
        mask_diffusion = tf.concat([
            tf.ones_like(eps[:, :, :self.num_diffusion], dtype=tf.bool),
            tf.zeros_like(eps[:, :, self.num_diffusion:], dtype=tf.bool)
        ], axis=-1)
        
        eps = tf.where(mask_diffusion, eps*x['input_mask'], tf.zeros_like(eps))
            
        t = tf.random.uniform((2*batch_size,1))                
        _, alpha, sigma = self.get_logsnr_alpha_sigma(t)
        
        perturbed_x = alpha*x['input_features'] + eps * sigma
        perturbed_x = tf.where(mask_diffusion,
                               perturbed_x*x['input_mask'],
                               tf.zeros_like(perturbed_x))
            
        v_pred_part = self.model_part([perturbed_x,
                                       perturbed_x[:,:,:2],
                                       x['input_mask'],
                                       x['input_jet'],t,y])
        
        v_pred_part = tf.reshape(v_pred_part[:,:,:self.num_diffusion],(tf.shape(v_pred_part)[0], -1))
        v_part = alpha * eps - sigma * x['input_features']
        v_part = tf.reshape(v_part[:,:,:self.num_diffusion],(tf.shape(v_part)[0], -1))
        loss_part = tf.reduce_sum(tf.square(v_part-v_pred_part))/(self.num_diffusion*tf.reduce_sum(x['input_mask']))
        loss = tf.reduce_mean(loss_jet) + tf.reduce_mean(loss_part)


        self.loss_tracker.update_state(loss)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)
            
        return {m.name: m.result() for m in self.metrics}
            
    def call(self,x):        
        return self.model(x)
    
    # def assign(self,):
    #     self.optimizer.build(self.model_part.trainable_variables + self.model_jet.trainable_variables)

    
    def generate(self,cond,jets=None,nsplit=2,jet_info=None):
        jet_info = []
        part_info = []

        if jets is not None:
            jet_split = np.array_split(jets,nsplit)
            
        for i,split in enumerate(np.array_split(cond,nsplit)):
            if jets is not None:
                dijet = jet_split[i]
            else:
                start = time.time()
                dijet = self.DDPMSampler(split,self.ema_jet,
                                         data_shape=[split.shape[0],2,self.num_jet],
                                         w = 0.0,
                                         num_steps = 1024,
                                         const_shape = self.shape).numpy()
                end = time.time()
                print("Time for sampling {} events is {} seconds".format(split.shape[0],end - start))
                
            jet_info.append(dijet)

            particles = []
            start = time.time()
            for ijet in range(2):                
                jet = dijet[:,ijet]
                nparts = np.expand_dims(np.clip(utils.revert_npart(jet[:,-1],str(self.max_part)),
                                                2,self.max_part),-1)
                #print(np.unique(nparts))
                mask = np.expand_dims(
                    np.tile(np.arange(self.max_part),(nparts.shape[0],1)) < np.tile(nparts,(1,self.max_part)),-1)
        
                assert np.sum(np.sum(mask.reshape(mask.shape[0],-1),-1,keepdims=True)-nparts)==0, 'ERROR: Particle mask does not match the expected number of particles'

                #start = time.time()
                parts = self.DDPMSampler(split,
                                         [self.ema_body,self.ema_head],
                                         data_shape=[split.shape[0],self.max_part,self.num_feat],
                                         jet=jet,
                                         const_shape = self.shape,
                                         mask=tf.convert_to_tensor(mask, dtype=tf.float32),
                                         num_steps = self.num_steps,
                                         w = 0.00,
                                         ).numpy()
                particles.append(parts*mask)
                
                # parts = np.ones(shape=(split.shape[0],self.max_part,3))
                # particles.append(parts)
                
            end = time.time()
            print("Time for sampling {} events is {} seconds".format(split.shape[0],end - start))
            part_info.append(np.stack(particles,1)) 
            
        return np.concatenate(part_info),np.concatenate(jet_info)


    def logsnr_schedule_cosine(self,t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))

    def get_logsnr_alpha_sigma(self,time):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
        
        return logsnr[:,None], alpha[:,None], sigma[:,None]

    def evaluate_models(self,head,body,x,jet,mask,t,cond,w = 0.0):
        
        x_in = mask*(x-self.mean)/self.std
        if w > 0.0:
            v = body([x_in,x[:,:,:2],mask,t], training=False)
            v = (1.0+w)*head([v,jet,mask,t,cond],training=False)  - w*head([v,jet,mask,t,tf.zeros_like(cond)],training=False)
        else:
            v = body([x_in,x[:,:,:2],mask,t], training=False)
            v = head([v,jet,mask,t,cond],training=False)  

        return mask*(self.std*v + self.mean)


    @tf.function
    def NoisySampler(self,
                     cond,
                     model,
                     data_shape=None,
                     const_shape=None,
                     num_steps=100,
                     jet=None,             
                     mask=None,
                     w=0):

        batch_size = cond.shape[0]
        x = self.prior_sde(data_shape)
        if jet is not None:
            mask_diffusion = tf.concat([
                tf.ones_like(x[:, :, :self.num_diffusion], dtype=tf.bool),
                tf.zeros_like(x[:, :, self.num_diffusion:], dtype=tf.bool)
            ], axis=-1)

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            logsnr_t, alpha, sigma = self.get_logsnr_alpha_sigma(t)
            logsnr_s, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps)

            if jet is None:
                v = model([x, t, cond], training=False)
                masks = None
            else:
                x = tf.where(mask_diffusion, x*mask, tf.zeros_like(x))                
                model_body, model_head = model
                v = self.evaluate_models(model_head,model_body,x,jet,mask,t,cond,w=w)
                masks = [mask,mask_diffusion]
            
            pred_x = alpha * x - sigma * v
            eps = v * alpha + x * sigma
            pred_x,_ = self.second_order_correction(t,x,pred_x,eps,
                                                    alpha,sigma,
                                                    cond,model,jet,masks,
                                                    num_steps=num_steps,
                                                    shape=const_shape
                                                    )

            
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
                                alphas,sigmas,
                                cond,model,jet=None,masks=None,
                                num_steps=100,
                                second_order_alpha=0.5,shape=None,w=0):
        step_size = 1.0/num_steps
        t = time_step - second_order_alpha * step_size
        _, alpha_signal_rates, alpha_noise_rates = self.get_logsnr_alpha_sigma(t)
        alpha_noisy_images = alpha_signal_rates * pred_images + alpha_noise_rates * pred_noises

        if jet is None:
            v = model([alpha_noisy_images, t,cond],training=False) 
        else:
            mask,mask_diffusion = masks
            alpha_noisy_images = tf.where(mask_diffusion, alpha_noisy_images*mask, tf.zeros_like(x))
            model_body, model_head = model
            v = self.evaluate_models(model_head,model_body,
                                     alpha_noisy_images,
                                     jet,mask,t,cond,w=w)
            
        alpha_pred_noises = alpha_noise_rates * alpha_noisy_images + alpha_signal_rates * v
        # linearly combine the two noise estimates
        pred_noises = (1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises + 1.0 / (
            2.0 * second_order_alpha
        ) * alpha_pred_noises

        mean = (x - sigmas * pred_noises) / alphas        
        eps = pred_noises
        
        return mean,eps

    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    const_shape=None,    
                    num_steps=100,
                    jet=None,
                    mask=None,w=0.):
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

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps            
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps)
            
            if jet is None:
                v = model([x, t, cond], training=False)
                masks = None
            else:
                x = tf.where(mask_diffusion, x*mask, tf.zeros_like(x))
                model_body, model_head = model
                v = self.evaluate_models(model_head,model_body,x,jet,mask,t,cond,w=w)
                masks = [mask,mask_diffusion]
                                                                                
            mean = alpha * x - sigma * v
            eps = v * alpha + x * sigma
                            
            mean,eps = self.second_order_correction(t,x,mean,eps,
                                                    alpha,sigma,
                                                    cond,model,jet,masks,
                                                    num_steps=num_steps,
                                                    shape=const_shape
                                                    )
            
            x = alpha_ * mean + sigma_ * eps
        return mean

    @tf.function
    def HeunSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    jet=None,
                    num_steps=100,
                    mask=None,w=0.,
                    logsnr_min=-20.,logsnr_max=20.):
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
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b

        if jet is not None:
            mask_diffusion = tf.concat([
                tf.ones_like(x[:, :, :self.num_diffusion], dtype=tf.bool),
                tf.zeros_like(x[:, :, self.num_diffusion:], dtype=tf.bool)
            ], axis=-1)

        for time_step in tf.range(num_steps, 0, delta=-1):
            t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / num_steps
            x_cur = x
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / num_steps)
            coeff = a*alpha/sigma
            coeff_ = a*alpha_/sigma_
            
            if jet is None:
                v = model([x, t, cond], training=False)
            else:
                x = tf.where(mask_diffusion, x*mask, tf.zeros_like(x))
                model_body, model_head = model
                v = self.evaluate_models(model_head,model_body,x,jet,mask,t,cond,w=w)

            mean = alpha * x - sigma * v
            d = coeff*x -coeff*mean
            x = x_cur - d/num_steps
            if time_step>1:
                if jet is None:
                    v = model([x, t - 1.0/num_steps, cond], training=False)
                else:
                    x = tf.where(mask_diffusion, x*mask, tf.zeros_like(x))
                    model_body, model_head = model
                    v = self.evaluate_models(model_head,model_body,x,jet,mask,t - 1.0/num_steps,cond,w=w)
                mean = alpha_ * x - sigma_ * v
                
                d_dash = coeff_*x -coeff_*mean
                x = x_cur - 0.5/num_steps*(d+d_dash)

        return x


class Classifier(keras.Model):
    """Score based generative model"""
    def __init__(self,
                 num_feat,
                 num_jet,      
                 num_classes=2,
                 projection_dim = 128,
                 local = True, K = 10,
                 num_local = 2, 
                 num_layers = 8, num_class_layers=2,
                 num_heads = 4,drop_probability = 0.0,
                 simple = False, layer_scale = True,
                 layer_scale_init = 1e-5,        
                 talking_head = False,
                 mode = 'classifier',
                 class_activation = None,
                 fine_tune = False,
                 model_name = None,):
        super(Classifier, self).__init__()


        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.class_activation = class_activation
        self.projection_dim = projection_dim
        self.layer_scale = layer_scale
        self.layer_scale_init=layer_scale_init
        self.num_diffusion=3

        self.model_part =  PET(num_feat=num_feat,
                               num_jet=num_jet,
                               num_classes=num_classes,
                               local = local,
                               num_layers = num_layers, 
                               drop_probability = drop_probability,
                               simple = simple, layer_scale = layer_scale,
                               talking_head = talking_head,
                               mode = mode,
                               class_activation=self.class_activation,
                               )
        
        if fine_tune:
            assert model_name is not None, "ERROR: Model name is necessary if fine tune is on"
            self.model_part.load_weights(model_name,by_name=True,skip_mismatch=True)

        #For this classifier we only need the body
        self.body = self.model_part.body
        self.head = self.model_part.classifier_head
        

        inputs_features = Input(shape=(2, None, num_feat),name='input_features')
        inputs_points = Input(shape=(2, None, 2),name='input_points')
        inputs_mask = Input((2,None,1),name='input_mask')
        inputs_time = Input((2,1),name='input_time')
        inputs_mass = Input((1),name='input_mass')
        inputs_jet = Input((2,self.num_jet),name='input_jet')

        npart = tf.shape(inputs_features)[2]
        
        #Flatten particles to preserve permutation equivariance inside jets
        inputs_reshape = tf.reshape(inputs_features,(-1,npart,tf.shape(inputs_features)[3]))
        points_reshape = tf.reshape(inputs_points,(-1,npart,2))
        mask_reshape = tf.reshape(inputs_mask,(-1,npart,1))
        time_reshape = tf.reshape(inputs_time,(-1,1))
               
        encoding_part = self.body([inputs_reshape,points_reshape,mask_reshape,time_reshape])
        
        encoding_part = tf.reshape(encoding_part,(-1,2,npart,self.projection_dim))
        encoding_jet = self.small_PET(inputs_jet)
        
        # encoding_jet = get_encoding(inputs_jet,2*self.projection_dim)
        encoding_mass = get_encoding(inputs_mass,2*self.projection_dim)
        # encoding_jet = encoding_jet + encoding_mass[:,None]
        
        scale,shift = tf.split(encoding_mass[:,None,None],2,-1)
        encoding_part = layers.GroupNormalization(groups=1)(encoding_part)*(1.0+scale) + shift        
        encoding_jet = layers.Dense(self.num_jet)(encoding_jet)

        
        outputs = self.head([tf.reshape(encoding_part,(-1,2*npart,self.projection_dim)),
                             layers.GlobalAvgPool1D()(encoding_jet)])
        

        self.model = keras.Model(inputs=[inputs_features,inputs_points,inputs_jet,
                                         inputs_mask,inputs_time,inputs_mass],
                                 outputs=outputs)

    
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
    def compile(self,body_optimizer,head_optimizer):
        super(Classifier, self).compile(experimental_run_tf_function=False,
                                        weighted_metrics=[],
                                        #run_eagerly=True
        )
        self.body_optimizer = body_optimizer
        self.optimizer = head_optimizer

    def small_PET(self,
                  input_features,
                  num_layers = 6,
                  ):

        encoded = get_encoding(input_features,self.projection_dim)
        for i in range(num_layers):
            x1 = layers.GroupNormalization(groups=1)(encoded)
            updates = layers.MultiHeadAttention(num_heads=self.num_heads,
                                                key_dim=self.projection_dim//self.num_heads)(x1,x1)
            updates = layers.GroupNormalization(groups=1)(updates)
            if self.layer_scale:
                updates = LayerScale(self.layer_scale_init, self.projection_dim)(updates)
            
            x2 = layers.Add()([updates,encoded])
            x3 = layers.GroupNormalization(groups=1)(x2)
            x3 = layers.Dense(4*self.projection_dim,activation="gelu")(x3)
            x3 = layers.Dense(self.projection_dim)(x3)
            if self.layer_scale:
                x3 = LayerScale(self.layer_scale_init, self.projection_dim)(x3)
            encoded = layers.Add()([x3,x2])

        encoded = layers.GroupNormalization(groups=1)(encoded)
        return encoded


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


    def train_step(self, inputs):
        x,y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        x['input_time'] = tf.zeros((batch_size,2,1))
        with tf.GradientTape(persistent=True) as tape:
            y_pred,y_mse = self.model(x)
            loss = weighted_binary_crossentropy(y, y_pred)
            #loss += mse(x['input_jet'][:,0]+x['input_jet'][:,1],y_mse)
            #loss = binary_crossentropy(y, y_pred,from_logits=True)


        self.body_optimizer.minimize(loss,self.body.trainable_variables,tape=tape)        
        self.optimizer.minimize(loss,self.head.trainable_variables,tape=tape)

        #self.optimizer.minimize(loss,self.model.trainable_variables,tape=tape)
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, inputs):
        x,y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        x['input_time'] = tf.zeros((batch_size,2,1))
        y_pred,_ = self.model(x)
        loss = weighted_binary_crossentropy(y, y_pred)
        #loss = binary_crossentropy(y, y_pred,from_logits=True)        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}




    
