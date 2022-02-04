import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import data
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
from data import image_read

py.arg('--experiment_dir', default='/home/Alexandrite/smin/cycle_git/data/pre_output/0202/2')
py.arg('--batch_size', type=int, default=1)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/cycle_git/data', 'knees', 'val_clean'), '*.png')
B_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/cycle_git/data', 'knees', 'val_noisy'), '*.png')

A_img_paths_test.sort()
B_img_paths_test.sort()

"""
A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)"""
B_dataset_test = data.make_dataset(B_img_paths_test, 1, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)

A_dataset_test = np.array(image_read(py.join('/home/Alexandrite/smin/cycle_git/data', 'knees', 'val_clean')), dtype=np.float32)

# model
#G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
#G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
#H = module.Extractor(input_shape=(args.crop_size, args.crop_size, 3))
G = module.Gen_with_adain()
H = module.Extractor()

# resotre
tl.Checkpoint(dict(G=G, H=H), py.join(args.experiment_dir, 'checkpoints')).restore()

"""
@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A
    
@tf.function
def sample_B2A(B):
    B2A = tf.clip_by_value(G_B2A(B, training=False), -1.0, 1.0)
    H2A = tf.clip_by_value(B - H(B, training=False), -1.0, 1.0)
    GnH = tf.clip_by_value(0.5*B2A + 0.5*H2A, -1.0, 1.0)

    return B2A, H2A, GnH
"""
tf.random.set_seed(5)
z = tf.random.normal([1,64], 0, 1, dtype=tf.float32)


@tf.function
def sample_B2A(B):
    B2A = tf.clip_by_value(G(B, training=False), -1.0, 1.0)
    H2A = tf.clip_by_value(B - H(B, training=False), -1.0, 1.0)
    GnH = tf.clip_by_value(0.5*B2A + 0.5*H2A, -1.0, 1.0)

    return B2A, H2A, GnH


# run
"""
save_dir = py.join(args.experiment_dir, 'samples_testing', 'A2B')
py.mkdir(save_dir)
i = 0
for A in A_dataset_test:
    A2B, A2B2A = sample_A2B(A)
    for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
        img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)
        im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
        i += 1
"""

psnr_tot = 0.0
ssim_tot = 0.0
psnr_result = []
ssim_result = []

h_psnr_tot = 0.0
h_ssim_tot = 0.0
h_psnr_result = []
h_ssim_result = []

e_psnr_tot = 0.0
e_ssim_tot = 0.0
e_psnr_result = []
e_ssim_result = []

save_dir = py.join(args.experiment_dir, 'samples_testing', 'B2A_knee')
py.mkdir(save_dir)
i = 0
j =0
for B in B_dataset_test:
    B2A, H2A, GnH = sample_B2A(B)
    tmp_A = A_dataset_test[i]
    for B_i, B2A_i, H2A_i, GnH_i in zip(B, B2A, H2A, GnH):
        B_i = B_i[:, :, 0]
        B2A_i = B2A_i[:, :, 0]
        H2A_i = H2A_i[:, :, 0]
        GnH_i = GnH_i[:, :, 0]
        tmp_A /= 255
        tmp_A = tmp_A * 2 -1
        img = np.concatenate([B_i.numpy(), B2A_i.numpy(), H2A_i.numpy(), GnH_i.numpy(), tmp_A], axis=1)
        im.imwrite(img, py.join(save_dir, "noise2clean_%d.jpg" % i ))

        tmp_A = (tmp_A + 1) * 0.5 * 255
        B2A = B2A[0, :, : , 0]
        H2A = H2A[0, :, :, 0]
        GnH = GnH[0, :, :, 0]
        tmp_B2A = (B2A + 1) * 0.5 * 255
        tmp_H2A = (H2A + 1) * 0.5 * 255
        tmp_GnH = (GnH + 1) * 0.5 * 255

        # img_psnr = psnr(tmp_A, tmp_B2A,  data_range=255.0)
        # img_ssim = ssim(tmp_A, tmp_B2A,  data_range=255.0)

        #im1 = tf.image.convert_image_dtype(tmp_A, tf.float32)
        #im2 = tf.image.convert_image_dtype(tmp_B2A, tf.float32)
        im1 = np.array(tmp_A, np.float32)
        im2 = np.array(tmp_B2A, np.float32)
        im3 = np.array(tmp_H2A, np.float32)
        im4 = np.array(tmp_GnH, np.float32)
        #img_psnr = psnr(im1, im2, max_val=255.0)
        #img_ssim = ssim(im1, im2, max_val=255.0)
        img_psnr = psnr(im1, im2,  data_range=255.0)
        img_ssim = ssim(im1, im2,  data_range=255.0)

        h_img_psnr = psnr(im1, im3, data_range=255.0)
        h_img_ssim = ssim(im1, im3, data_range=255.0)

        e_img_psnr = psnr(im1, im4, data_range=255.0)
        e_img_ssim = ssim(im1, im4, data_range=255.0)


        psnr_tot += img_psnr
        ssim_tot += img_ssim

        h_psnr_tot += h_img_psnr
        h_ssim_tot += h_img_ssim

        e_psnr_tot += e_img_psnr
        e_ssim_tot += e_img_ssim
        j += 1
    i += 1

print(i,j)


psnr_result.append(psnr_tot / 100)
ssim_result.append(ssim_tot / 100)

h_psnr_result.append(h_psnr_tot / 100)
h_ssim_result.append(h_ssim_tot / 100)

e_psnr_result.append(e_psnr_tot / 100)
e_ssim_result.append(e_ssim_tot / 100)

        #### psnr 계산
#print("input brain PSNR: %f, SSIM %f" % (sum(input_psnr), sum(input_ssim)))
print("result knee PSNR: %f, SSIM %f" % (sum(psnr_result), sum(ssim_result)))
print("h result knee PSNR: %f, SSIM %f" % (sum(h_psnr_result), sum(h_ssim_result)))
print("e result knee PSNR: %f, SSIM %f" % (sum(e_psnr_result), sum(e_ssim_result)))


#print("psnr: ", img_psnr)
#print("ssim: ", img_ssim)


