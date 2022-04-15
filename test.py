import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl
import h5py

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import data
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
from data import image_read

py.arg('--experiment_dir', default='/home/Alexandrite/smin/cycle_git/data/output/0317/6/')
py.arg('--batch_size', type=int, default=1)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
"""
A_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/cycle_git/data/brain', 'db_valid'), '*.png')
B_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/cycle_git/data/knees', 'val_noisy'), '*.png')
"""
"""
A_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI/ours', 'brain', 'clean_R6_6_val'), '*.png')
B_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI/ours', 'brain', 'noisy_R6_6_val'), '*.png')
"""
A_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee/multicoil_val', 'clean_R4_8'), '*.png')
B_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee/multicoil_val', 'noisy_R4_8'), '*.png')
A_img_paths_test = A_img_paths_test[-100:]
B_img_paths_test = B_img_paths_test[-100:]

A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
B_dataset_test = data.make_dataset(B_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)


#A_dataset_test = np.array(image_read(py.join('/home/Alexandrite/smin/cycle_git/data/knees', 'val_clean')), dtype=np.float32)



# model
G_A2B = module.ResnetGenerator()
G_B2A = module.ResnetGenerator()

# resotre
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return B2A, B2A2B


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

psnr_input = 0.0
ssim_input = 0.0
psnr_in = []
ssim_in = []

save_dir = py.join(args.experiment_dir, 'samples_testing', 'B2A')
py.mkdir(save_dir)
i = 0
for A, B in zip(A_dataset_test, B_dataset_test):
    B2A, B2A2B = sample_B2A(B)
    for A_i, B_i, B2A_i, B2A2B_i in zip(A, B, B2A, B2A2B):
        B_i = B_i[:, :, 0]
        B2A_i = B2A_i[: ,:, 0]
        A_i = A_i[:,:,0]
        img = np.concatenate([B_i.numpy(), B2A_i.numpy(), A_i], axis=1)
        im.imwrite(img, py.join(save_dir, "noise2clean_%d.jpg" % i ))
        i += 1

        A = A[0, :, :, 0]
        tmp_A = (A + 1) * 0.5 * 255
        B2A = B2A[0, :, : , 0]
        tmp_B2A = (B2A + 1) * 0.5 * 255

        B = B[0, :, :, 0]
        tmp_B = (B + 1) * 0.5 * 255
        # img_psnr = psnr(tmp_A, tmp_B2A,  data_range=255.0)
        # img_ssim = ssim(tmp_A, tmp_B2A,  data_range=255.0)

        #im1 = tf.image.convert_image_dtype(tmp_A, tf.float32)
        #im2 = tf.image.convert_image_dtype(tmp_B2A, tf.float32)
        im1 = np.array(tmp_A, np.float32)
        im2 = np.array(tmp_B2A, np.float32)
        im3 = np.array(tmp_B, np.float32)
        im_psnr = psnr(im1, im3,data_range=255.0)
        im_ssim = ssim(im1, im3, data_range=255.0)
        img_psnr = psnr(im1, im2,  data_range=255.0)
        img_ssim = ssim(im1, im2,  data_range=255.0)


        psnr_tot += img_psnr
        ssim_tot += img_ssim
        psnr_input += im_psnr
        ssim_input += im_ssim

len_dataset = 100

psnr_result.append(psnr_tot / len_dataset)
ssim_result.append(ssim_tot / len_dataset)
psnr_in.append(psnr_input / len_dataset)
ssim_in.append(ssim_input / len_dataset)


        #### psnr 계산
#print("input brain PSNR: %f, SSIM %f" % (sum(input_psnr), sum(input_ssim)))
print("input PSNR: %f, SSIM: %f " % (sum(psnr_in), sum(ssim_in)))
print("result brain PSNR: %f, SSIM %f" % (sum(psnr_result), sum(ssim_result)))

#print("psnr: ", img_psnr)
#print("ssim: ", img_ssim)


