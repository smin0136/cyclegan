
from PIL import Image
import tensorflow as tf
import imlib as im

import numpy as np

def tf_complex(data):
    real = data[0, ...]
    imag = data[1, ...]

    del data
    data = tf.complex(real, imag)
    data = tf.identity(data)
    return data

def tf_channel(data):
    real = tf.math.real(data)
    imag = tf.math.imag(data)
    real = real[tf.newaxis, ...]
    imag = imag[tf.newaxis, ...]
    del data
    data = tf.concat([real, imag], axis=0)
    data = tf.identity(data)

    return data

def tf_RF(image, mask):
    image = tf_complex(image)
    mask = tf_complex(mask)

    freq_full = tf.signal.fft2d(image) ## 이거
    freq_zero = tf.zeros_like(freq_full)
    condition = tf.cast(tf.math.real(mask)>0.9, tf.bool) ## 이거

    freq_dest = tf.where(condition, freq_full, freq_zero, name='RfFf')

    #mask = mask[0]
    #freq_dest = freq_full * mask + 0.0

    freq_dest = tf_channel(freq_dest)

    return tf.identity(freq_dest)

def tf_FhRh(freq, mask):
    # shape b,2,h,w,1
    freq = tf_complex(freq)
    mask = tf_complex(mask)
    # shape b,h,w,1

    #condition = tf.cast((tf.math.real(mask)>0.9), tf.bool)
    freq_full = freq
    #freq_zero = tf.zeros_like(freq_full)
    #freq_dest = tf.where(condition, freq_full, freq_zero)
    image = tf.signal.ifft2d(freq_full)  # b,h,w,1

    image = tf_channel(image) # b,2,h,w,1
    return image


def sampling_matrix(clean, mask):

    #img = Image.open("/home/Alexandrite/smin/cycle_git/data/mask/radial/mask_1/mask_10.png")
    """
    img = Image.open(mask)
    mask = np.array(img, dtype=np.float32)
    mask /= 255
    mask = tf.convert_to_tensor(mask)
    """
    clean = (clean + 1)*0.5
    mask = mask[..., 0]
    mask_real = mask[tf.newaxis, ...]
    mask_imag = tf.zeros_like(mask)[tf.newaxis, ...]
    mask_temp = tf.concat([mask_real, mask_imag], axis=0)

    clean_t = clean[0]
    clean_t = clean_t[..., 0]
    clean_real = clean_t[tf.newaxis, ...]  # 1.64.64
    clean_imag = tf.zeros_like(clean_t)[tf.newaxis, ...]

    clean_temp = tf.concat([clean_real, clean_imag], axis=0)

    #mask_temp = tf_channel(mask)
    #clean_temp = tf_channel(clean_t)

    Mf1 = tf_RF(clean_temp, mask_temp)
    #print("Mf1", Mf1.shape)
    #Mf1 = tf_FhRh(Mf1, mask_temp)
    #print("Mf2", Mf1.shape)

    #tf.print(tf.math.reduce_max(Mf1))
    #print(Mf1[0].shape)

    return Mf1[0]