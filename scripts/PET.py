import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from layers import StochasticDepth, TalkingHeadAttention, LayerScale, RandomDrop
from tensorflow.keras.losses import mse, categorical_crossentropy


class PET(keras.Model):
    """Point-Edge Transformer"""
    def __init__(self,
                 num_feat,
                 num_jet,      
                 num_classes=2,
                 num_keep = 7, #Number of features that wont be dropped
                 feature_drop = 0.1,
                 projection_dim = 128,
                 local = True, K = 10,
                 num_local = 2, 
                 num_layers = 8, num_class_layers=2,
                 num_gen_layers = 2,
                 num_heads = 4,drop_probability = 0.0,
                 simple = False, layer_scale = True,
                 layer_scale_init = 1e-5,        
                 talking_head = False,
                 mode = 'classifier',
                 num_diffusion = 3,
                 dropout=0.0,
                 class_activation=None,
                 ):

        super(PET, self).__init__()
        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_classes = num_classes
        self.num_keep = num_keep
        self.feature_drop = feature_drop
        self.drop_probability = drop_probability
        self.dropout = dropout
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.layer_scale_init=layer_scale_init
        self.mode = mode
        self.num_diffusion = num_diffusion
        self.ema=0.999
        self.class_activation = class_activation
        
        
        input_features = layers.Input(shape=(None, num_feat), name='input_features')
        input_points = layers.Input(shape=(None, 2), name='input_points')
        input_mask = layers.Input((None,1),name = 'input_mask')
        input_jet = layers.Input((num_jet),name='input_jet')
        input_label = layers.Input((num_classes),name='input_label')
        input_time = layers.Input((1),name = 'input_time')


        outputs_body = self.PET_body(input_features,
                                     input_points,
                                     input_mask,
                                     input_time,
                                     local = local, K = K,
                                     num_local = num_local, 
                                     talking_head = talking_head)

        self.body = keras.Model(inputs=[input_features,input_points,input_mask,input_time],
                                outputs=outputs_body)
        

        outputs_classifier,outputs_regressor = self.PET_classifier(outputs_body,
                                                                   input_jet,
                                                                   num_class_layers=num_class_layers,
                                                                   num_jet=num_jet,
                                                                   simple = simple
                                                                   )
        
        outputs_generator = self.PET_generator(outputs_body,
                                               input_jet,
                                               label=input_label,
                                               time=input_time,
                                               mask=input_mask,
                                               num_layers=num_gen_layers,
                                               simple=simple,
                                               )

        self.classifier_head = keras.Model(inputs=[outputs_body,input_jet],
                                           outputs=[outputs_classifier,outputs_regressor])
        self.generator_head = keras.Model(inputs=[outputs_body,input_jet,
                                                  input_mask,input_time,input_label],
                                          outputs=outputs_generator)
        
        self.classifier = keras.Model(inputs=[input_features,input_points,input_mask,
                                              input_jet,input_time],
                                      outputs=[outputs_classifier,outputs_regressor])
        self.generator = keras.Model(inputs=[input_features,input_points,input_mask,
                                             input_jet,input_time,input_label],
                                     outputs=outputs_generator)
        self.ema_body = keras.models.clone_model(self.body)
        self.ema_generator_head = keras.models.clone_model(self.generator_head)


        
        self.pred_tracker = keras.metrics.CategoricalAccuracy(name="acc")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_tracker = keras.metrics.Mean(name="mse")
        self.gen_tracker = keras.metrics.Mean(name="score")
        self.pred_smear_tracker = keras.metrics.CategoricalAccuracy(name="smear_acc")
        self.mse_smear_tracker = keras.metrics.Mean(name="smear_mse")
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        metrics = [self.loss_tracker]
        if 'all' in self.mode or self.mode == 'classifier':
            metrics.append(self.pred_tracker)
        if  'all' in self.mode or self.mode == 'generator':
            metrics.append(self.gen_tracker)
        if self.mode == 'all':
            metrics.append(self.mse_tracker)
            metrics.append(self.mse_smear_tracker)
            metrics.append(self.pred_smear_tracker)
        return metrics


    def call(self,x):
        if self.mode == 'generator':
            return self.generator(x)
        else:
            return self.classifier(x)

    def train_step(self, inputs):
        x,y = inputs
        batch_size = tf.shape(x['input_jet'])[0]
        x['input_time'] = tf.zeros((batch_size,1))
        with tf.GradientTape(persistent=True) as tape:
            loss = 0.0            
            if self.mode == 'classifier' or 'all' in self.mode:
                body = self.body(x)
                            
            if self.mode == 'classifier' or 'all' in self.mode:
                y_pred,y_mse = self.classifier_head([body,x['input_jet']])
                loss_pred = categorical_crossentropy(y, y_pred,from_logits=True)
                loss += loss_pred
                if 'all' in self.mode:    
                    loss_mse = mse(x['input_jet'],y_mse)
                    loss += loss_mse
                
            if self.mode == 'generator' or 'all' in self.mode:
                t = tf.random.uniform((batch_size,1))                
                _, alpha, sigma = get_logsnr_alpha_sigma(t)

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
                
                perturbed_body = self.body([
                    perturbed_x,
                    perturbed_x[:,:,:2],
                    x['input_mask'],t])
                
                v_pred = self.generator_head([
                    perturbed_body,x['input_jet'],x['input_mask'],t,y])
                v_pred = tf.reshape(v_pred[:,:,:self.num_diffusion],(tf.shape(v_pred)[0], -1))
                
                v = alpha[:,None] * eps - sigma[:,None] * x['input_features']
                v = tf.reshape(v[:,:,:self.num_diffusion],(tf.shape(v)[0], -1))
                loss_part = tf.reduce_sum(tf.square(v-v_pred))/(self.num_diffusion*tf.reduce_sum(x['input_mask']))
                loss += loss_part

            if self.mode == 'all':
                #Add a classification task over perturbed inputs
                y_pred_smear, y_mse_smear = self.classifier_head([perturbed_body,x['input_jet']])
                loss_pred_smear = categorical_crossentropy(y, y_pred_smear,from_logits=True)
                loss += alpha**2*loss_pred_smear
                                                                       
                loss_mse_smear =  mse(x['input_jet'],y_mse_smear)
                loss += alpha**2*loss_mse_smear
                

        self.loss_tracker.update_state(loss)
        
        if self.mode == 'classifier':
            trainable_vars = self.classifier_head.trainable_variables
            self.pred_tracker.update_state(y, y_pred)

            
        if self.mode == 'generator':
            trainable_vars = self.generator_head.trainable_variables
            self.gen_tracker.update_state(loss_part)
            
        if self.mode == 'all':
            trainable_vars = self.classifier_head.trainable_variables + self.generator_head.trainable_variables
            self.pred_tracker.update_state(y, y_pred)
            self.gen_tracker.update_state(loss_part)
            self.mse_tracker.update_state(loss_mse)
            self.mse_smear_tracker.update_state(loss_mse_smear)
            self.pred_smear_tracker(y,y_pred_smear)
            self.mse_tracker.update_state(loss_mse)
            
        if self.mode == 'all_min':
            trainable_vars = self.classifier_head.trainable_variables + self.generator_head.trainable_variables
            self.gen_tracker.update_state(loss_part)           

        
        self.body_optimizer.minimize(loss,self.body.trainable_variables,tape=tape)
        self.optimizer.minimize(loss,trainable_vars,tape=tape)


        
        for weight, ema_weight in zip(self.body.weights, self.ema_body.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        for weight, ema_weight in zip(self.generator_head.weights, self.ema_generator_head.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
            
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, inputs):
        x,y = inputs
        loss = 0.0
        batch_size = tf.shape(x['input_jet'])[0]
        x['input_time'] = tf.zeros((batch_size,1))

        if self.mode == 'classifier' or 'all' in self.mode:
            body = self.body(x)
                    
        if self.mode == 'classifier' or 'all' in self.mode:
            y_pred,y_mse = self.classifier_head([body,x['input_jet']])
            loss_pred = categorical_crossentropy(y, y_pred,from_logits=True)
            loss += loss_pred
            if 'all' in self.mode:
                loss_mse = mse(x['input_jet'],y_mse)
                loss += loss_mse

        if self.mode == 'generator' or 'all' in self.mode:
            t = tf.random.uniform((batch_size,1))                
            _, alpha, sigma = get_logsnr_alpha_sigma(t)

            eps = tf.random.normal((tf.shape(x['input_features'])),
                                 dtype=tf.float32)*x['input_mask'][:,:,None]
            mask_diffusion = tf.concat([
                tf.ones_like(eps[:, :, :self.num_diffusion], dtype=tf.bool),
                tf.zeros_like(eps[:, :, self.num_diffusion:], dtype=tf.bool)
            ], axis=-1)
            eps = tf.where(mask_diffusion, eps, tf.zeros_like(eps))
            perturbed_x = alpha[:,None]*x['input_features'] + eps * sigma[:,None]
            perturbed_x = tf.where(mask_diffusion, perturbed_x, tf.zeros_like(perturbed_x))
            perturbed_body = self.body([
                perturbed_x,
                perturbed_x[:,:,:2],
                x['input_mask'],t])

                
            v_pred = self.generator_head([
                    perturbed_body,x['input_jet'],x['input_mask'],t,y])
        
            v_pred = tf.reshape(v_pred[:,:,:self.num_diffusion],(tf.shape(v_pred)[0], -1))
                
            v = alpha[:,None] * eps - sigma[:,None] * x['input_features']
            v = tf.reshape(v[:,:,:self.num_diffusion],(tf.shape(v)[0], -1))
            loss_part = tf.reduce_sum(tf.square(v-v_pred))/(self.num_diffusion*tf.reduce_sum(x['input_mask']))
            #loss_part = mse(v_pred,v)            
            loss += loss_part


        if self.mode == 'all':
            #Add a classification task over perturbed inputs
            y_pred_smear,y_mse_smear = self.classifier_head([perturbed_body,x['input_jet']])
            loss_pred_smear = categorical_crossentropy(y, y_pred_smear,from_logits=True)
            loss += loss_pred_smear
                                                       
            loss_mse_smear =  mse(x['input_jet'],y_mse_smear)
                
            loss += loss_mse_smear
            

        self.loss_tracker.update_state(loss)        
        if self.mode == 'classifier' or 'all' in self.mode:
            self.pred_tracker.update_state(y, y_pred)
            
        if self.mode == 'generator' or 'all' in self.mode:
            self.gen_tracker.update_state(loss_part)
        if  self.mode == 'all':            
            self.mse_smear_tracker.update_state(loss_mse_smear)
            self.pred_smear_tracker(y,y_pred_smear)
            self.mse_tracker.update_state(loss_mse)
            
        return {m.name: m.result() for m in self.metrics}
                
    def PET_body(self,
                 input_features,
                 input_points,
                 input_mask,
                 input_time,
                 local, K,num_local,
                 talking_head,
                 ):



        
        #Randomly drop features not present in other datasets
        encoded  = RandomDrop(self.feature_drop if  'all' in self.mode else 0.0,num_skip=self.num_keep)(input_features)                        
        encoded = get_encoding(encoded,self.projection_dim)

        time = FourierProjection(input_time,self.projection_dim)
        time = tf.tile(time[:,None, :], [1,tf.shape(encoded)[1], 1])*input_mask
        time = layers.Dense(2*self.projection_dim,activation='gelu',use_bias=False)(time)
        scale,shift = tf.split(time,2,-1)
        
        encoded = encoded*(1.0+scale) + shift

        
        if local:
            coord_shift = tf.multiply(999., tf.cast(tf.equal(input_mask, 0), dtype='float32'))        
            points = input_points[:,:,:2]
            local_features = input_features
            for _ in range(num_local):
                local_features = get_neighbors(coord_shift+points,local_features,self.projection_dim,K)
                points = local_features
                
            encoded = layers.Add()([local_features,encoded])

        skip_connection = encoded
        for i in range(self.num_layers):
            x1 = layers.GroupNormalization(groups=1)(encoded)
            if talking_head:
                updates, _ = TalkingHeadAttention(self.projection_dim, self.num_heads, 0.0)(x1)
            else:
                updates = layers.MultiHeadAttention(num_heads=self.num_heads,
                                                    key_dim=self.projection_dim//self.num_heads)(x1,x1)

            if self.layer_scale:
                updates = LayerScale(self.layer_scale_init, self.projection_dim)(updates,input_mask)
            updates = StochasticDepth(self.drop_probability)(updates)
            
            x2 = layers.Add()([updates,encoded])
            x3 = layers.GroupNormalization(groups=1)(x2)
            x3 = layers.Dense(2*self.projection_dim,activation="gelu")(x3)
            x3 = layers.Dropout(self.dropout)(x3)
            x3 = layers.Dense(self.projection_dim)(x3)
            if self.layer_scale:
                x3 = LayerScale(self.layer_scale_init, self.projection_dim)(x3,input_mask)
            x3 = StochasticDepth(self.drop_probability)(x3)
            encoded = layers.Add()([x3,x2])*input_mask
                    
        return encoded + skip_connection


    def compile(self,body_optimizer,head_optimizer):
        super(PET, self).compile(experimental_run_tf_function=False,
                                  weighted_metrics=[],
                                  #run_eagerly=True
                                  )
        self.body_optimizer = body_optimizer
        self.optimizer = head_optimizer

        
    def PET_classifier(
            self,
            encoded,
            input_jet,
            num_class_layers,
            num_jet,
            simple = False
    ):

        #Include event information as a representative particle
        if simple:
            encoded = layers.GroupNormalization(groups=1)(encoded)
            representation = layers.GlobalAveragePooling1D()(encoded)
            jet_encoded = get_encoding(input_jet,self.projection_dim)
            representation = layers.Dense(self.projection_dim,activation='gelu')(representation+jet_encoded)
            outputs_pred = layers.Dense(self.num_classes,activation=self.class_activation)(representation)
            outputs_mse = layers.Dense(num_jet)(representation)
        else:
            conditional = layers.Dense(2*self.projection_dim,activation='gelu')(input_jet)
            conditional = tf.tile(conditional[:,None, :], [1,tf.shape(encoded)[1], 1])
            scale,shift = tf.split(conditional,2,-1)
            encoded = encoded*(1.0 + scale) + shift

            class_tokens = tf.Variable(tf.zeros(shape=(1, self.projection_dim)),trainable = True)    
            class_tokens = tf.tile(class_tokens[None, :, :], [tf.shape(encoded)[0], 1, 1])
                        
            for _ in range(num_class_layers):
                concatenated = tf.concat([class_tokens, encoded],1)

                x1 = layers.GroupNormalization(groups=1)(concatenated)            
                updates = layers.MultiHeadAttention(num_heads=self.num_heads,
                                                    key_dim=self.projection_dim//self.num_heads)(
                                                        query=x1[:,:1], value=x1, key=x1)
                updates = layers.GroupNormalization(groups=1)(updates)
                if self.layer_scale:
                    updates = LayerScale(self.layer_scale_init, self.projection_dim)(updates)

                x2 = layers.Add()([updates,class_tokens])
                x3 = layers.GroupNormalization(groups=1)(x2)
                x3 = layers.Dense(2*self.projection_dim,activation="gelu")(x3)
                x3 = layers.Dropout(self.dropout)(x3)
                x3 = layers.Dense(self.projection_dim)(x3)
                if self.layer_scale:
                    x3 = LayerScale(self.layer_scale_init, self.projection_dim)(x3)
                class_tokens = layers.Add()([x3,x2])


            class_tokens = layers.GroupNormalization(groups=1)(class_tokens)
            outputs_pred = layers.Dense(self.num_classes,activation=self.class_activation)(class_tokens[:,0])
            outputs_mse = layers.Dense(num_jet)(class_tokens[:,0])

        return outputs_pred,outputs_mse


    def PET_generator(
            self,
            encoded,
            input_jet,
            label,            
            time,
            mask,
            num_layers,
            simple
    ):


        time = FourierProjection(time,self.projection_dim)
        cond_jet = layers.Dense(self.projection_dim,activation="gelu")(input_jet)        
        cond_token = layers.Dense(2*self.projection_dim,activation="gelu")(tf.concat([time,cond_jet],-1))
        cond_token = layers.Dense(self.projection_dim,activation="gelu")(cond_token)



        cond_label = layers.Dense(self.projection_dim,use_bias=False)(label)
        cond_label = StochasticDepth(self.feature_drop)(cond_label)
        cond_token = layers.Add()([cond_token,cond_label])

        cond_token = tf.tile(cond_token[:,None, :], [1,tf.shape(encoded)[1], 1])*mask

        if simple:
            cond_token = layers.Dense(2*self.projection_dim,activation='gelu')(cond_token)
            scale,shift = tf.split(cond_token,2,-1)        
            encoded = layers.GroupNormalization(groups=1)(encoded)*(1.0+scale) + shift
            encoded = layers.Dense(2*self.projection_dim,activation='gelu')(encoded)
            encoded = layers.Dropout(self.dropout)(encoded)
            encoded = layers.Dense(self.num_feat)(encoded)*mask

        else:
            for _ in range(num_layers):
                concatenated = layers.Add()([cond_token , encoded])
                x1 = layers.GroupNormalization(groups=1)(concatenated)

                updates = layers.MultiHeadAttention(num_heads=self.num_heads,
                                                    key_dim=self.projection_dim//self.num_heads)(
                                                        query=x1, value=x1, key=x1)
                
                if self.layer_scale:
                    updates = LayerScale(self.layer_scale_init, self.projection_dim)(updates,mask)

                x2 = layers.Add()([updates,cond_token])
                x3 = layers.GroupNormalization(groups=1)(x2)
                x3 = layers.Dense(2*self.projection_dim,activation="gelu")(x3)
                x3 = layers.Dense(self.projection_dim)(x3)
                if self.layer_scale:
                    x3 = LayerScale(self.layer_scale_init, self.projection_dim)(x3,mask)
                cond_token = layers.Add()([x3,x2])

            encoded = layers.GroupNormalization(groups=1)(cond_token+encoded)            
            encoded = layers.Dense(self.num_feat)(encoded)*mask
        
        return encoded


def get_neighbors(points,features,projection_dim,K):
    drij = pairwise_distance(points)  # (N, P, P)
    _, indices = tf.nn.top_k(-drij, k=K + 1)  # (N, P, K+1)
    indices = indices[:, :, 1:]  # (N, P, K)
    knn_fts = knn(tf.shape(points)[1], K, indices, features)  # (N, P, K, C)
    knn_fts_center = tf.broadcast_to(tf.expand_dims(features, 2), tf.shape(knn_fts))
    local = tf.concat([knn_fts-knn_fts_center,knn_fts_center],-1)
    local = layers.Dense(2*projection_dim,activation='gelu')(local)
    local = layers.Dense(projection_dim,activation='gelu')(local)
    local = tf.reduce_mean(local,-2)
    
    return local


def pairwise_distance(point_cloud):
    r = tf.reduce_sum(point_cloud * point_cloud, axis=2, keepdims=True)
    m = tf.matmul(point_cloud, point_cloud, transpose_b = True)
    D = r - 2 * m + tf.transpose(r, perm=(0, 2, 1)) + 1e-5
    return D


def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)    
    batch_size = tf.shape(features)[0]

    batch_indices = tf.reshape(tf.range(batch_size), (-1, 1, 1))
    batch_indices = tf.tile(batch_indices, (1, num_points, k))
    indices = tf.stack([batch_indices, topk_indices], axis=-1)
    return tf.gather_nd(features, indices)


def get_encoding(x,projection_dim,use_bias=True):
    x = layers.Dense(2*projection_dim,use_bias=use_bias,activation='gelu')(x)
    x = layers.Dense(projection_dim,use_bias=use_bias,activation='gelu')(x)
    return x

def FourierProjection(x,projection_dim,num_embed=64):    
    half_dim = num_embed // 2
    emb = tf.math.log(10000.0) / (half_dim - 1)
    emb = tf.cast(emb,tf.float32)
    freq = tf.exp(-emb* tf.range(start=0, limit=half_dim, dtype=tf.float32))


    angle = x*freq*1000.0
    embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)*x
    embedding = layers.Dense(2*projection_dim,activation="swish",use_bias=False)(embedding)
    embedding = layers.Dense(projection_dim,activation="swish",use_bias=False)(embedding)
    
    return embedding


def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
    b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
    a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
    return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))
    
def get_logsnr_alpha_sigma(time):
    logsnr = logsnr_schedule_cosine(time)
    alpha = tf.sqrt(tf.math.sigmoid(logsnr))
    sigma = tf.sqrt(tf.math.sigmoid(-logsnr))        
    return logsnr, alpha, sigma
