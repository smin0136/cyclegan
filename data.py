import numpy as np
import tensorflow as tf
import tf2lib as tl
import os
from PIL import Image





def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            #img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [load_size, load_size])
            #img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])

            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1

            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size, crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)



def image_read(path):
    output = []
    if ('.tif' not in path) and ('.png' not in path):
        tif_list = sorted(os.listdir(path))
        file_list_py = [file for file in tif_list if file.endswith(".png")]

        #tif_list = os.listdir(path)
        print(file_list_py)
        for name in file_list_py:
            im = Image.open(path+"/"+name)
            output.append(np.array(im, dtype=np.float32))
    else:
        im = Image.open(path)
        output = np.array(im, dtype=np.float32)
    return output


def image_augmentation(x):
    assert (x.ndim == 4 or x.ndim == 3), ("Check the dimension of inputs")
    if x.ndim==3:
        n,w,h = np.shape(x)[0:3]
        out = np.zeros([n*8,w,h], dtype=np.float32)
        for f in range(0,2):
            for r in range(0,4):
                if f == 0 and r == 0:
                    out[:n] = x
                    continue
                for i in range(0,n):
                    out[i+(n*r)+f*4*n] = np.flip(np.rot90(x[i],r),f)
    elif x.ndim==4:
        n,w,h,z = np.shape(x)[0:4]
        out = np.zeros([n*8,w,h,z], dtype=np.float32)
        for f in range(0,2):
            for r in range(0,4):
                if f == 0 and r == 0:
                    out[:n] = x
                    continue
                for i in range(0,n):
                    for j in range(0,z):
                        out[i+(n*r)+f*4*n,:,:,j] = np.flip(np.rot90(x[i,:,:,j],r),f)
    return out


def image_division(image, patch_size):
    assert (image.ndim <= 4 and image.ndim >= 3), ("Check the dimension of inputs")  # n, h, w or n, h, w, 1
    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)

    n = len(image)
    patch_x, patch_y = patch_size
    output = []
    for i in range(0, n):
        temp = image[i]
        x, y, z = np.shape(temp)
        p = int(np.ceil(x / patch_x))
        q = int(np.ceil(y / patch_y))
        for j in range(0, p):
            for k in range(0, q):
                if j == p - 1:
                    if k == q - 1:
                        output.append(temp[-patch_x:, -patch_y:, 0:z])
                    else:
                        output.append(temp[-patch_x:, k * patch_y:(k + 1) * patch_y, 0:z])
                else:
                    if k == q - 1:
                        output.append(temp[j * patch_x:(j + 1) * patch_x, -patch_y:, 0:z])
                    else:
                        output.append(temp[j * patch_x:(j + 1) * patch_x, k * patch_y:(k + 1) * patch_y, 0:z])

    return np.array(output, dtype=np.float32)