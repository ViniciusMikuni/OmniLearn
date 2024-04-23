import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class StochasticDepth(layers.Layer):
    """Stochastic Depth layer (https://arxiv.org/abs/1603.09382).

    Reference:
        https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(
                shape, minval=0, maxval=1)
            random_tensor = tf.floor(random_tensor)
            return x * random_tensor
        
        return x


class RandomDrop(layers.Layer):
    def __init__(self, drop_prob: float, num_skip: float, **kwargs):
        super(RandomDrop, self).__init__(**kwargs)
        self.drop_prob = drop_prob
        self.num_skip = num_skip


    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],1) 
            random_tensor = keep_prob + tf.random.uniform(
                shape, minval=0, maxval=1)
            random_tensor = tf.floor(random_tensor)
            x[:,:,self.num_skip:] = x[:,:,self.num_skip:] * random_tensor[:,None]
            return x
        
        return x

    
class SimpleHeadAttention(layers.Layer):
    """Simple MHA where masks can be directly added to the inputs.
    Args:
        projection_dim (int): projection dimension for the query, key, and value
            of attention.
        num_heads (int): number of attention heads.
        dropout_rate (float): dropout rate to be used for dropout in the attention
            scores as well as the final projected outputs.
    """
    def __init__(
        self, projection_dim: int, num_heads: int, dropout_rate: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        
        head_dim = self.projection_dim // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = layers.Dense(projection_dim * 3)
        self.proj = layers.Dense(projection_dim)
        self.proj_drop = layers.Dropout(dropout_rate)
        self.softmax = layers.Softmax(axis=-1)

    def call(self, x,int_matrix = None,mask=None, training=False):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        # Project the inputs all at once.
        qkv = self.qkv(x)


        # Reshape the projected output so that they're segregated in terms of
        # query, key, and value projections.
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))

        # Transpose so that the `num_heads` becomes the leading dimensions.
        # Helps to better segregate the representation sub-spaces.
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        scale = tf.cast(self.scale, dtype=qkv.dtype)
        q, k, v = qkv[0] * scale, qkv[1], qkv[2]

        # Obtain the raw attention scores.
        attn = tf.matmul(q, k, transpose_b = True)
        
        # Normalize the attention scores.

        if int_matrix is not None:
            attn+=int_matrix

        if mask is not None:
            mask = tf.cast(mask, dtype=attn.dtype)
            mask = tf.tile(mask, [1, tf.shape(attn)[1], 1, 1])
            attn += (1.0-mask)*-1e9

        attn = self.softmax(attn)

        # Final set of projections as done in the vanilla attention mechanism.
        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))
        
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x, attn

class TalkingHeadAttention(layers.Layer):
    """Talking-head attention as proposed in CaiT: https://arxiv.org/abs/2003.02436.
    Args:
        projection_dim (int): projection dimension for the query, key, and value
            of attention.
        num_heads (int): number of attention heads.
        dropout_rate (float): dropout rate to be used for dropout in the attention
            scores as well as the final projected outputs.
    """
    def __init__(
        self, projection_dim: int, num_heads: int, dropout_rate: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        
        head_dim = self.projection_dim // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = layers.Dense(projection_dim * 3)
        self.attn_drop = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(projection_dim)
        self.proj_l = layers.Dense(self.num_heads)
        self.proj_w = layers.Dense(self.num_heads)
        self.proj_drop = layers.Dropout(dropout_rate)
        self.softmax = layers.Softmax(axis=-1)

    def call(self, x,int_matrix = None,mask=None, training=False):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        # Project the inputs all at once.
        qkv = self.qkv(x)


        # Reshape the projected output so that they're segregated in terms of
        # query, key, and value projections.
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))

        # Transpose so that the `num_heads` becomes the leading dimensions.
        # Helps to better segregate the representation sub-spaces.
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        scale = tf.cast(self.scale, dtype=qkv.dtype)
        q, k, v = qkv[0] * scale, qkv[1], qkv[2]

        # Obtain the raw attention scores.
        attn = tf.matmul(q, k, transpose_b = True)
        if int_matrix is not None:
            attn+=int_matrix


        # Linear projection of the similarities between the query and key projections.
        attn = self.proj_l(tf.transpose(attn, perm=[0, 2, 3, 1]))
        
        # Normalize the attention scores.
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        
        if mask is not None:
            mask = tf.cast(mask, dtype=attn.dtype)
            mask = tf.tile(mask, [1, tf.shape(attn)[1], 1, 1])
            attn += (1.0-mask)*-1e9

        attn = self.softmax(attn)
                
        
        # Linear projection on the softmaxed scores.
        attn = self.proj_w(tf.transpose(attn, perm=[0, 2, 3, 1]))
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        attn = self.attn_drop(attn, training=training)

        # Final set of projections as done in the vanilla attention mechanism.
        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))
        
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x, attn


class LayerScale(layers.Layer):
    def __init__(self, init_values, projection_dim, **kwargs):
        super(LayerScale, self).__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
        self.gamma_initializer = tf.keras.initializers.Constant(self.init_values)

    def build(self, input_shape):
        # Ensure the layer is properly built by defining its weights in the build method
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=self.gamma_initializer,
            trainable=True,
            name='gamma'
        )
        super(LayerScale, self).build(input_shape)

    def call(self, inputs,mask=None):
        # Element-wise multiplication of inputs and gamma
        if mask is not None:
            return inputs * self.gamma* mask
        else:
            return inputs * self.gamma

