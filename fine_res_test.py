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

py.arg('--experiment_dir', default='/home/Alexandrite/smin/cycle_git/data/output/0414/2/')
py.arg('--batch_size', type=int, default=1)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    test                                    =s
# ==============================================================================

# data
"""
A_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/cycle_git/data/brain', 'clean_R4_8_val'), '*.png')
B_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/cycle_git/data/brain', 'noisy_R4_8_val'), '*.png')"""


A_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI/ours', 'brain', 'clean_R4_8_val'), '*.png')
B_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI/ours', 'brain', 'noisy_R4_8_val'), '*.png')

"""
A_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee/singlecoil_val', 'clean_R4_8'), '*.png')
B_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee/singlecoil_val', 'noisy_R4_8'), '*.png')
A_img_paths_test = A_img_paths_test[-100:]
B_img_paths_test = B_img_paths_test[-100:]
"""

A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
B_dataset_test = data.make_dataset(B_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)

# model


G = module.Gen_with_adain()
H = module.Extractor()

#G_A2B = module.ResnetGenerator()
#G_B2A = module.ResnetGenerator()

#H = module.Extractor ()


# resotre
#tl.Checkpoint(dict(G_A2B=G_A2B,G_B2A=G_B2A, H=H), py.join(args.experiment_dir, 'checkpoints')).restore()
tl.Checkpoint(dict(G=G, H=H), py.join(args.experiment_dir, 'checkpoints')).restore()


"""
@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A
"""

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

in_psnr_tot = 0.0
in_ssim_tot = 0.0
in_psnr_result = []
in_ssim_result = []

save_dir = py.join(args.experiment_dir, 'samples_testing', '50')
py.mkdir(save_dir)
i = 0
for A, B in zip(A_dataset_test, B_dataset_test):
    B2A, H2A, GnH = sample_B2A(B)
    for A_i, B_i, B2A_i, H2A_i, GnH_i in zip(A, B, B2A, H2A, GnH):
        B_i = B_i[:, :, 0]
        B2A_i = B2A_i[: ,:, 0]
        H2A_i = H2A_i[:, :, 0]
        GnH_i = GnH_i[:, :, 0]
        A_i = A_i[:, :, 0]
        img = np.concatenate([B_i.numpy(), B2A_i.numpy(), H2A_i.numpy(), GnH_i.numpy(), A_i.numpy()], axis=1)
        im.imwrite(img, py.join(save_dir, "noise2clean_%d.jpg" % i ))

        A = A[0, :, :, 0]
        B = B[0, :, :, 0]
        B2A = B2A[0, :, :, 0]
        H2A = H2A[0, :, :, 0]
        GnH = GnH[0, :, :, 0]
        tmp_A = (A + 1) * 0.5 * 255
        tmp_B = (B + 1) * 0.5 * 255
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
        im5 = np.array(tmp_B, np.float32)
        #img_psnr = psnr(im1, im2, max_val=255.0)
        #img_ssim = ssim(im1, im2, max_val=255.0)
        img_psnr = psnr(im1, im2,  data_range=255.0)
        img_ssim = ssim(im1, im2,  data_range=255.0)

        h_img_psnr = psnr(im1, im3, data_range=255.0)
        h_img_ssim = ssim(im1, im3, data_range=255.0)

        e_img_psnr = psnr(im1, im4, data_range=255.0)
        e_img_ssim = ssim(im1, im4, data_range=255.0)

        in_img_psnr = psnr(im1, im5, data_range=255.0)
        in_img_ssim = ssim(im1, im5, data_range=255.0)


        psnr_tot += img_psnr
        ssim_tot += img_ssim

        h_psnr_tot += h_img_psnr
        h_ssim_tot += h_img_ssim

        e_psnr_tot += e_img_psnr
        e_ssim_tot += e_img_ssim

        in_psnr_tot += in_img_psnr
        in_ssim_tot += in_img_ssim
    i += 1

len_dataset = 100

psnr_result.append(psnr_tot / len_dataset)
ssim_result.append(ssim_tot / len_dataset)

h_psnr_result.append(h_psnr_tot / len_dataset)
h_ssim_result.append(h_ssim_tot / len_dataset)

e_psnr_result.append(e_psnr_tot / len_dataset)
e_ssim_result.append(e_ssim_tot / len_dataset)

in_psnr_result.append(in_psnr_tot / len_dataset)
in_ssim_result.append(in_ssim_tot / len_dataset)



        #### psnr 계산
print("input PSNR: %f, SSIM %f" % (sum(in_psnr_result), sum(in_ssim_result)))
print("result PSNR: %f, SSIM %f" % (sum(psnr_result), sum(ssim_result)))
print("h result PSNR: %f, SSIM %f" % (sum(h_psnr_result), sum(h_ssim_result)))
print("e result PSNR: %f, SSIM %f" % (sum(e_psnr_result), sum(e_ssim_result)))

#print("psnr: ", img_psnr)
#print("ssim: ", img_ssim)


