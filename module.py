import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras import layers



# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization

"""
def ResnetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)

def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h) # 128 128
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h) # 64 -> 32
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    #h : shared features
    # Head_contrastive = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h) 32 -> 16
    # Head_contrastive = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(Head_contrastive) 16 -> 8,8,ch
    '''
    
    features_normalized = tf.math.l2_normalize(features, axis=1) # batch_size, 65536
    logits = tf.divide(
        tf.linalg.matmul(features_normalized, tf.transpose(features_normalized)), self.temperature 
    ) # batch_size, batch_size -> 모든 경우의 수에 대한 cosine similarity 가 계산된 행렬
    
    ## Hard Contrastive Regularization
    #y_true = tf.linalg.matmul(label,label, transpose_b=True, a_is_sparse=True, b_is_sparse=True)
    # y_true (?) class 가 
    
    y_true = tf.where(y_true > self.T, tf.ones(tf.shape(y_true), dtype=tf.float32), tf.zeros(tf.shape(y_true), dtype=tf.float32))
    contrastive_loss = self.alpha*tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
    '''
    dim = min(dim * 2, dim_ * 8)
    h = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h) # 32 -> 32
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2) #

    # 3
    h = keras.layers.Conv2D(1, 4, strides=1, padding='same')(h) # 32 -> 32



    return keras.Model(inputs=inputs, outputs=h)


def Extractor(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_blocks=20,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = keras.layers.Conv2D(dim, (3,3), (1,1), activation=tf.nn.leaky_relu, padding="same", use_bias=True)(h)

    # 2
    for _ in range(n_blocks-2):
        h = keras.layers.Conv2D(dim, (3,3), (1,1), activation=None, padding="same", use_bias=True)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h)

    # 3
    h = keras.layers.Conv2D(output_channels, (3,3), (1,1), activation=None, padding="same", use_bias=True)(h)

    return keras.Model(inputs=inputs, outputs=h)
"""

class Res_Block(keras.Model):
    def __init__(self, dim=64):
        super(Res_Block, self).__init__()
        self.n1 = tfa.layers.InstanceNormalization()
        self.n2 = tfa.layers.InstanceNormalization()
        self.dim = dim

        self.h1 = layers.Conv2D(dim, 3, padding='valid', use_bias=False)
        self.h2 = layers.Conv2D(dim, 3, padding='valid', use_bias=False)

    def __call__(self, inputs, training=True):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h1(x)
        x = self.n1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h2(x)
        x = self.n2(x, training=training)
        out = layers.add([inputs, x])

        return out


class ResnetGenerator(keras.Model):
    def __init__(self, output_channels=3, dim=64, n_downsamplings=2, n_blocks=9, norm='instance_norm'):
        super(ResnetGenerator, self).__init__()
        self.start_neuron = 3
        self.dim = dim
        self.n_downsamplings = n_downsamplings
        self.n_blocks = n_blocks
        self.n1 = tfa.layers.InstanceNormalization()
        self.n2 = tfa.layers.InstanceNormalization()
        self.n3 = tfa.layers.InstanceNormalization()
        self.n4 = tfa.layers.InstanceNormalization()
        self.n5 = tfa.layers.InstanceNormalization()


        self.cv1 = layers.Conv2D(dim, 7, padding='valid', use_bias=False)
        self.cv2 = layers.Conv2D(dim*2, 3, strides=2, padding='same', use_bias=False)
        self.cv3 = layers.Conv2D(dim*4, 3, strides=2, padding='same', use_bias=False)

        self.res = []
        for i in range(0, n_downsamplings):
            self.res.append(Res_Block(dim*4))

        self.cv4 = layers.Conv2DTranspose(dim*2, 3, strides=2, padding='same', use_bias=False)
        self.cv5 = layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)
        self.out = layers.Conv2D(output_channels, 7, padding='valid')

    def __call__(self, inputs, z=None, training=True):
        x = inputs
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.cv1(x)
        x = self.n1(x)
        x = tf.nn.relu(x)

        x = self.cv2(x)
        x = self.n2(x)
        x = tf.nn.relu(x)

        x = self.cv3(x)
        x = self.n3(x)
        x = tf.nn.relu(x)

        for h1 in self.res:
            x = h1(x, training=training)

        x = self.cv4(x)
        x = self.n4(x)
        x = tf.nn.relu(x)

        x = self.cv5(x)
        x = self.n5(x)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.out(x)
        x = tf.tanh(x)

        return x

"""
class ResnetGenerator_adain(keras.Model):
    def __init__(self, output_channels=3, dim=64, n_downsamplings=2, n_blocks=9, norm='instance_norm'):
        super(ResnetGenerator_adain, self).__init__()
        self.start_neuron = 3
        self.initializer = 'truncated_normal'
        self.dim = dim
        self.n_downsamplings = n_downsamplings
        self.n_blocks = n_blocks
        self.AIN = AdaIN(tf.nn.leaky_relu)
        self.n1 = tfa.layers.InstanceNormalization()
        self.n2 = tfa.layers.InstanceNormalization()
        self.n3 = tfa.layers.InstanceNormalization()

        self.cv1 = layers.Conv2D(dim, 7, padding='valid', use_bias=False)
        self.cv2 = layers.Conv2D(dim*2, 3, strides=2, padding='same', use_bias=False)
        self.cv3 = layers.Conv2D(dim*4, 3, strides=2, padding='same', use_bias=False)

        self.res = []
        for i in range(0, n_blocks):
            self.res.append(Res_Block(dim*4))

        #res1
        self.h1 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)
        self.h2 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)
        #res2
        self.h3 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)
        self.h4 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)

        self.h5 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)
        self.h6 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)

        self.h7 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)
        self.h8 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)

        self.h9 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)
        self.h10 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)

        self.h11 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)
        self.h12 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)

        self.h13 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)
        self.h14 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)

        self.h15 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)
        self.h16 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)

        self.h17 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)
        self.h18 = keras.layers.Conv2D(dim*4, 3, padding='valid', use_bias=False)

        ##########


        self.cv4 = layers.Conv2DTranspose(dim*2, 3, strides=2, padding='same', use_bias=False)
        self.cv5 = layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)
        self.out = layers.Conv2D(output_channels, 7, padding='valid')

        self.d1 = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d2 = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d3 = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d4 = layers.Dense(self.dim, kernel_initializer=self.initializer)


        self.d5_m = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d5_v = layers.Dense(self.dim, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d6_m = layers.Dense(self.dim * 2, kernel_initializer=self.initializer)
        self.d6_v = layers.Dense(self.dim * 2, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d7_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d7_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)

        self.d8_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d8_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d9_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d9_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d10_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d10_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d11_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d11_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d12_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d12_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d13_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d13_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d14_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d14_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d15_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d15_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d16_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d16_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d17_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d17_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d18_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d18_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d19_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d19_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d20_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d20_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d21_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d21_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d22_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d22_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d23_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d23_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d24_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d24_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d25_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d25_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)


        self.d26_m = layers.Dense(self.dim * 2, kernel_initializer=self.initializer)
        self.d26_v = layers.Dense(self.dim * 2, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d27_m = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d27_v = layers.Dense(self.dim, activation=tf.nn.relu, kernel_initializer=self.initializer)

    def __call__(self, inputs, z=None, training=True):
        if z is not None:
            l = self.d1(z)
            l = self.d2(l)
            l = self.d3(l)
            latent = self.d4(l)

        x = inputs
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.cv1(x)

        if z is not None:
            x = self.AIN(x, self.d5_m(latent), self.d5_v(latent))
        else:
            x = self.AIN(x)

        #x = self.n1(x)
        #x = tf.nn.relu(x)
        ##### d layers dimension 확인 필요 ///  residaul block 은 어떻게 처리??

        x = self.cv2(x)
        if z is not None:
            x = self.AIN(x, self.d6_m(latent), self.d6_v(latent))
        else:
            x = self.AIN(x)
        #x = self.n2(x)
        #x = tf.nn.relu(x)


        x = self.cv3(x)
        if z is not None:
            x = self.AIN(x, self.d7_m(latent), self.d7_v(latent))
        else:
            x = self.AIN(x)
        #x = self.n3(x)
        #x = tf.nn.relu(x)


        ############### res
        skip_x = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h1(x)
        if z is not None:
            x = self.AIN(x, self.d8_m(latent), self.d8_v(latent))
        else:
            x = self.AIN(x)
        #x = self.n1(x, training=training)
        #x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h2(x)
        if z is not None:
            x = self.AIN(x, self.d9_m(latent), self.d9_v(latent), act=False)
        else:
            x = self.AIN(x, act=False)
        #x = self.n2(x, training=training)
        x = layers.add([skip_x, x])
        ######## 1
        skip_x = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h3(x)
        if z is not None:
            x = self.AIN(x, self.d10_m(latent), self.d10_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n1(x, training=training)
        # x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h4(x)
        if z is not None:
            x = self.AIN(x, self.d11_m(latent), self.d11_v(latent), act=False)
        else:
            x = self.AIN(x, act=False)
        # x = self.n2(x, training=training)
        x = layers.add([skip_x, x])
        ######## 2

        skip_x = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h5(x)
        if z is not None:
            x = self.AIN(x, self.d12_m(latent), self.d12_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n1(x, training=training)
        # x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h6(x)
        if z is not None:
            x = self.AIN(x, self.d13_m(latent), self.d13_v(latent), act=False)
        else:
            x = self.AIN(x, act=False)
        # x = self.n2(x, training=training)
        x = layers.add([skip_x, x])
        ######## 3

        skip_x = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h7(x)
        if z is not None:
            x = self.AIN(x, self.d14_m(latent), self.d14_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n1(x, training=training)
        # x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h8(x)
        if z is not None:
            x = self.AIN(x, self.d15_m(latent), self.d15_v(latent), act=False)
        else:
            x = self.AIN(x, act=False)
        # x = self.n2(x, training=training)
        x = layers.add([skip_x, x])
        ######## 4


        skip_x = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h9(x)
        if z is not None:
            x = self.AIN(x, self.d16_m(latent), self.d16_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n1(x, training=training)
        # x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h10(x)
        if z is not None:
            x = self.AIN(x, self.d17_m(latent), self.d17_v(latent), act=False)
        else:
            x = self.AIN(x, act=False)
        # x = self.n2(x, training=training)
        x = layers.add([skip_x, x])
        ######## 5
        
        skip_x = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h11(x)
        if z is not None:
            x = self.AIN(x, self.d18_m(latent), self.d18_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n1(x, training=training)
        # x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h12(x)
        if z is not None:
            x = self.AIN(x, self.d19_m(latent), self.d19_v(latent), act=False)
        else:
            x = self.AIN(x, act=False)
        # x = self.n2(x, training=training)
        x = layers.add([skip_x, x])
        ######## 6


        skip_x = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h13(x)
        if z is not None:
            x = self.AIN(x, self.d20_m(latent), self.d20_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n1(x, training=training)
        # x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h14(x)
        if z is not None:
            x = self.AIN(x, self.d21_m(latent), self.d21_v(latent), act=False)
        else:
            x = self.AIN(x, act=False)
        # x = self.n2(x, training=training)
        x = layers.add([skip_x, x])
        
        skip_x = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h15(x)
        if z is not None:
            x = self.AIN(x, self.d22_m(latent), self.d22_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n1(x, training=training)
        # x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h16(x)
        if z is not None:
            x = self.AIN(x, self.d23_m(latent), self.d23_v(latent), act=False)
        else:
            x = self.AIN(x, act=False)
        # x = self.n2(x, training=training)
        x = layers.add([skip_x, x])

        skip_x = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h17(x)
        if z is not None:
            x = self.AIN(x, self.d24_m(latent), self.d24_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n1(x, training=training)
        # x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = self.h18(x)
        if z is not None:
            x = self.AIN(x, self.d25_m(latent), self.d25_v(latent), act=False)
        else:
            x = self.AIN(x, act=False)
        # x = self.n2(x, training=training)
        x = layers.add([skip_x, x])


        ##################################
        x = self.cv4(x)
        if z is not None:
            x = self.AIN(x, self.d26_m(latent), self.d26_v(latent))
        else:
            x = self.AIN(x)
        #x = self.n4(x)
        #x = tf.nn.relu(x)

        x = self.cv5(x)
        if z is not None:
            x = self.AIN(x, self.d27_m(latent), self.d27_v(latent))
        else:
            x = self.AIN(x)
        #x = self.n5(x)
        #x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.out(x)
        x = tf.tanh(x)

        return x
"""


class Gen_with_adain(keras.Model):
    def __init__(self, output_channels=3, dim=64, n_downsamplings=2, n_blocks=9, norm='instance_norm'):
        super(Gen_with_adain, self).__init__()
        self.start_neuron = 3
        self.initializer = 'truncated_normal'
        self.dim = dim
        self.n_downsamplings = n_downsamplings
        self.n_blocks = n_blocks
        self.AIN = AdaIN(tf.nn.leaky_relu)

        self.cv1 = layers.Conv2D(dim, 7, padding='valid', use_bias=False)
        self.cv2 = layers.Conv2D(dim * 2, 3, strides=2, padding='same', use_bias=False)
        self.cv3 = layers.Conv2D(dim * 4, 3, strides=2, padding='same', use_bias=False)

        self.res = []
        for i in range(0, n_blocks):
            self.res.append(Res_Block(dim * 4))

        self.cv4 = layers.Conv2DTranspose(dim * 2, 3, strides=2, padding='same', use_bias=False)
        self.cv5 = layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)
        self.out = layers.Conv2D(output_channels, 7, padding='valid')

        self.d1 = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d2 = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d3 = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d4 = layers.Dense(self.dim, kernel_initializer=self.initializer)

        self.d5_m = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d5_v = layers.Dense(self.dim, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d6_m = layers.Dense(self.dim * 2, kernel_initializer=self.initializer)
        self.d6_v = layers.Dense(self.dim * 2, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d7_m = layers.Dense(self.dim * 4, kernel_initializer=self.initializer)
        self.d7_v = layers.Dense(self.dim * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)

        self.d8_m = layers.Dense(self.dim * 2, kernel_initializer=self.initializer)
        self.d8_v = layers.Dense(self.dim * 2, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.d9_m = layers.Dense(self.dim, kernel_initializer=self.initializer)
        self.d9_v = layers.Dense(self.dim, activation=tf.nn.relu, kernel_initializer=self.initializer)


    def __call__(self, inputs, z=None, training=True):
        if z is not None:
            l = self.d1(z)
            l = self.d2(l)
            l = self.d3(l)
            latent = self.d4(l)

        x = inputs
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.cv1(x)

        if z is not None:
            x = self.AIN(x, self.d5_m(latent), self.d5_v(latent))
        else:
            x = self.AIN(x)

        # x = self.n1(x)
        # x = tf.nn.relu(x)
        ##### d layers dimension 확인 필요 ///  residaul block 은 어떻게 처리??

        x = self.cv2(x)
        if z is not None:
            x = self.AIN(x, self.d6_m(latent), self.d6_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n2(x)
        # x = tf.nn.relu(x)

        x = self.cv3(x)
        if z is not None:
            x = self.AIN(x, self.d7_m(latent), self.d7_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n3(x)
        # x = tf.nn.relu(x)

        for h1 in self.res:
            x = h1(x, training=training)

        x = self.cv4(x)
        if z is not None:
            x = self.AIN(x, self.d8_m(latent), self.d8_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n4(x)
        # x = tf.nn.relu(x)

        x = self.cv5(x)
        if z is not None:
            x = self.AIN(x, self.d9_m(latent), self.d9_v(latent))
        else:
            x = self.AIN(x)
        # x = self.n5(x)
        # x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.out(x)
        x = tf.tanh(x)

        return x


class AdaIN(keras.layers.Layer):
    def __init__(self, activation):
        super(AdaIN, self).__init__()
        self.eps = 1e-7
        self.activation = layers.Activation(activation=activation)

    def __call__(self, input, y_mean=None, y_var=None, act=True):
        x_mean, x_var = tf.nn.moments(input, [1,2], keepdims=True) # N, 1, 1, C
        if y_mean is not None:
            y_mean = y_mean[:, tf.newaxis, tf.newaxis, :]
            y_var = y_var[:, tf.newaxis, tf.newaxis, :]

            x = (input-x_mean)/(tf.sqrt(x_var+self.eps))
            x = x*tf.sqrt(y_var+self.eps)+y_mean
        else:
            x = (input-x_mean)/(tf.sqrt(x_var+self.eps))
        return self.activation(x) if act else x


class ConvDiscriminator(keras.Model):
    def __init__(self, output_channels=3, dim=64, n_downsamplings=3, norm='instance_norm'):
        super(ConvDiscriminator, self).__init__()
        self.output_channel = output_channels
        self.dim = dim
        self.n_downsampling = n_downsamplings
        self.norm = norm

        self.n1 = tfa.layers.InstanceNormalization()
        self.n2 = tfa.layers.InstanceNormalization()
        self.n3 = tfa.layers.InstanceNormalization()

        self.cv1 = keras.layers.Conv2D(dim, 4, strides=2, padding='same')
        dim = min(dim * 2, self.dim * 8)
        self.cv2 = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)
        dim = min(dim * 2, self.dim * 8)
        self.cv3 = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)
        dim = min(dim * 2, self.dim * 8)
        self.cv4 = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)
        self.out = keras.layers.Conv2D(1, 4, strides=1, padding='same')

    def __call__(self, inputs, training=True):
        x = self.cv1(inputs)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.cv2(x)
        x = self.n1(x, training=training)
        x = tf.nn.leaky_relu(x, aplha=0.2)
        x = self.cv3(x)
        x = self.n2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.cv4(x)
        x = self.n3(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.out(x)

        return x


class ConvDiscriminator_cont(keras.Model):
    def __init__(self, output_channels=3, dim=64, n_downsamplings=3, norm='instance_norm'):
        super(ConvDiscriminator_cont, self).__init__()
        self.output_channel = output_channels
        self.dim = dim
        self.n_downsampling = n_downsamplings
        self.norm = norm

        self.n1 = tfa.layers.InstanceNormalization()
        self.n2 = tfa.layers.InstanceNormalization()
        self.n3 = tfa.layers.InstanceNormalization()

        self.cv1 = keras.layers.Conv2D(dim, 4, strides=2, padding='same')
        dim = min(dim * 2, self.dim * 8)
        self.cv2 = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)
        dim = min(dim * 2, self.dim * 8)
        self.cv3 = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)
        dim = min(dim * 2, self.dim * 8)
        self.cv4 = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)
        self.out = keras.layers.Conv2D(1, 4, strides=1, padding='same')

        self.h1 = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)
        self.h2 = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)

        # h : shared features
        # Head_contrastive = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h) 32 -> 16
        # Head_contrastive = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(Head_contrastive) 16 -> 8,8,ch


    def __call__(self, inputs, training=True):
        x = self.cv1(inputs)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.cv2(x)
        x = self.n1(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.cv3(x)
        x = self.n2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        h = self.h1(x)
        h = self.h2(h)


        x = self.cv4(x)
        x = self.n3(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.out(x)

        return x, h

class Extractor(keras.Model):
    def __init__(self, output_channels=3, dim=64, n_blocks=3, norm='instance_norm'):
        super(Extractor, self).__init__()
        self.output_channels = output_channels
        self.dim = dim
        self.n_blocks = n_blocks
        self.norm = norm

        self.cv1 = layers.Conv2D(dim, (3,3), (1,1), activation=tf.nn.leaky_relu, padding="same", use_bias=True)

        self.res = []
        self.nor =[]
        for _ in range(n_blocks-2):
            self.res.append(keras.layers.Conv2D(dim, (3, 3), (1, 1), activation=None, padding="same", use_bias=True))
            self.nor.append(tfa.layers.InstanceNormalization())

        self.out = layers.Conv2D(output_channels, (3,3), (1,1), activation=None, padding="same", use_bias=True)

    def __call__(self, inputs, training=True):
        x = self.cv1(inputs)
        for conv, nor in zip(self.res, self.nor):
            x = conv(x)
            x = nor(x, training=training)
            x = tf.nn.leaky_relu(x)
        x = self.out(x)

        return x

# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
