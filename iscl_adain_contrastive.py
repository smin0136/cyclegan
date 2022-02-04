import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='brain')
py.arg('--datasets_dir', default='/home/Alexandrite/smin/cycle_git/data')
py.arg('--load_size', type=int, default=256)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay', type=int, default=50)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--temperature', type=float, default=1.0)  # pool size to store fake samples
py.arg('--cl_weight', type=float, default=1.0)  # pool size to store fake samples
py.arg('--output_date', default='0110')
py.arg('--dir_num', default='1')
py.arg('--experiment_dir')
args = py.args()

# output_dir
output_dir = py.join(args.datasets_dir, 'output', args.output_date, args.dir_num)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'db_train'), '*.png')
B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'train_noisy'), '*.png')
A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, training=True, repeat=False)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'db_valid'), '*.png')
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'noisy'), '*.png')


B_img_paths_test.sort()



A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size, args.crop_size, shuffle=False, training=False, repeat=True)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G = module.Gen_with_adain()

D_A = module.ConvDiscriminator_cont()
Head_A = module.Projection_head()
D_B = module.ConvDiscriminator_cont()
Head_B = module.Projection_head()

H = module.Extractor()

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()
mae_loss_fn = tf.losses.MeanAbsoluteError()
contrastive_loss_fn = gan.SupervisedContrastiveLoss(args.temperature, args.batch_size)

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
H_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)
H_optimizer = keras.optimizers.Adam(learning_rate=H_lr_scheduler, beta_1=args.beta_1)


#pre_output_dir = py.join(args.datasets_dir, 'pre_output', args.output_date, args.dir_num)
pre_output_dir = py.join(args.datasets_dir, 'pre_output', '0202', '4')
#py.mkdir(output_dir)


############## 만약 pre training 이 H 학습시킨거면 H추가해야함#####################################################
tl.Checkpoint(dict(G=G, H=H), py.join(pre_output_dir, 'checkpoints')).restore()



# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

tf.random.set_seed(5)
z = tf.random.normal([1,64], 0, 1, dtype=tf.float32)


@tf.function
def train_G(A, B):
    with tf.GradientTape() as t:
        #z1 = z+tf.random.normal(tf.shape(z), mean=0.0, stddev=1.0, dtype=tf.float32)*1e-5
        z1=z
        A2B = G(A, z=z1,training=True)
        B2A = G(B, training=True)
        #############################  B-B2A <-> H(B)
        A2B2A = G(A2B, training=True)
        B2A2B = G(B2A, z=z1, training=True)
        #A2A = G(A,  training=True)
        #B2B = G(B, z=z1,training=True)

        A2B_d_logits, _ = D_B(A2B, training=True)
        B2A_d_logits, _ = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        #A2A_id_loss = identity_loss_fn(A, A2A)
        #B2B_id_loss = identity_loss_fn(B, B2B)

        ### bypass loss
        clean_H = B - H(B, training=False)
        noisy_H = A + H(B, training=False)
        y_hat_j = G(noisy_H, training=True)

        bypass_loss = tf.reduce_mean(tf.abs(B2A - clean_H)) + tf.reduce_mean(tf.abs(A - y_hat_j))

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss + bypass_loss) * args.cycle_loss_weight # + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

    G_grad = t.gradient(G_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'bypass_loss': bypass_loss}


@tf.function
def train_D(A, B, A2B, B2A):
    with tf.GradientTape() as t:
        A_d_logits, real_clean = D_A(A, training=True)
        B2A_d_logits, fake_clean = D_A(B2A, training=True)
        B_d_logits, real_noisy = D_B(B, training=True)
        A2B_d_logits, fake_noisy = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        A_d_cl = contrastive_loss_fn(Head_A(real_clean), Head_A(fake_clean))
        B_d_cl = contrastive_loss_fn(Head_B(real_noisy), Head_B(fake_noisy))
        sc_loss = cl_weight*(A_d_cl + B_d_cl)
        D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        """
        ################################# 요런식?
        temperature = 0.07
        T = 0.9
        alpha = 1

        features_normalized = tf.math.l2_normalize(real_noisy, axis=1)  # batch_size, 65536
        logits = tf.divide(
            tf.linalg.matmul(features_normalized, tf.transpose(features_normalized)), temperature
        )  # batch_size, batch_size -> 모든 경우의 수에 대한 cosine similarity 가 계산된 행렬

        ## Hard Contrastive Regularization
        y_true = tf.linalg.matmul(A, A, transpose_b=True, a_is_sparse=True, b_is_sparse=True)
        # y_true (?) class 가

        y_true = tf.where(y_true > T, tf.ones(tf.shape(y_true), dtype=tf.float32),tf.zeros(tf.shape(y_true), dtype=tf.float32))

        contrastive_loss = alpha * tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
        #########################################"""

        ######## bst loss
        clean_H = B - H(B, training=False)
        noisy_H = A + H(B, training=False)
        fake_clean, _ = D_A(clean_H, training=True)
        fake_noisy, _ = D_B(noisy_H, training=True)

        bst_loss = tf.reduce_mean(tf.math.square(fake_noisy)) + tf.reduce_mean(tf.math.square(fake_clean))

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * args.gradient_penalty_weight + bst_loss + args.cl_weight*sc_loss 


    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables + Head_A.trainable_variables + Head_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables + Head_A.trainable_variables + Head_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'bst_loss': bst_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp,
            'CL_A' : A_d_cl,
            'CL_B' : B_d_cl,
            }


@tf.function
def train_H(A,B):
    with tf.GradientTape() as t:
        #z1 = z+tf.random.normal(tf.shape(z), mean=0.0, stddev=1.0, dtype=tf.float32)*1e-5
        z1 = z
        n_hat_i = H(B, training=True) # A가 noisy B가 clean noise
        n_bar_i = B - G(B, training=True) # nosiy - clean noise
        x_hat_j = G(A,  z=z1, training=True) # fake noisy
        n_tilda_j = H(x_hat_j, training=True) # fake noisy noise

        pseudo_loss = tf.reduce_mean(tf.abs(n_hat_i - n_bar_i))
        noise_consistency = tf.reduce_mean(tf.abs(x_hat_j - A - n_tilda_j))
        loss = pseudo_loss + noise_consistency

    H_grad = t.gradient(loss, H.trainable_variables)
    H_optimizer.apply_gradients(zip(H_grad, H.trainable_variables))

    return {'pseudo_loss': pseudo_loss,
            'noise_consistency': noise_consistency}


def train_step(A, B):
    A2B, B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A)
    H_loss_dict = train_H(A, B)

    return G_loss_dict, D_loss_dict, H_loss_dict


@tf.function
def sample(A, B):
    A2B = tf.clip_by_value(G(A, z=z, training=False), -1.0, 1.0)
    B2A = tf.clip_by_value(G(B, training=False), -1.0, 1.0)
    H2A = B - H(B, training=False)
    print(tf.math.reduce_max(H2A))
    H2A = tf.clip_by_value(H2A,-1.0, 1.0)
    GnH = tf.clip_by_value(0.5 * B2A + 0.5 * H2A, -1.0, 1.0)
    #A2B2A = G_B2A(A2B, training=False)
    #B2A2B = G_A2B(B2A, training=False)
    return A2B, B2A, H2A, GnH


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G=G,
                                D_A=D_A,
                                D_B=D_B,
                                H = H,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                H_optimizer=H_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)

"""
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)
"""

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)


# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)
        # train for an epoch

        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict, H_loss_dict = train_step(A, B)


            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=D_optimizer.iterations, name='D_losses')
            tl.summary(H_loss_dict, step=H_optimizer.iterations, name='H_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                A, B = next(test_iter)
                if A is None or B is None :
                    continue

                A2B, B2A, H2A, GnH = sample(A, B)
                img = im.immerge(np.concatenate([A, A2B, H2A, B, B2A, GnH], axis=0), n_rows=2)
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

                # psnr ssim 계산

                tmp_A = (A + 1)*0.5*255
                tmp_B2A = (B2A + 1)*0.5*255
                tmp_H2A = (H2A + 1)*0.5*255
                #img_psnr = psnr(tmp_A, tmp_B2A,  data_range=255.0)
                #img_ssim = ssim(tmp_A, tmp_B2A,  data_range=255.0)

                im1 = tf.image.convert_image_dtype(tmp_A, tf.float32)
                im2 = tf.image.convert_image_dtype(tmp_B2A, tf.float32)
                im3 = tf.image.convert_image_dtype(tmp_H2A, tf.float32)
                img_psnr = tf.image.psnr(im1, im2, max_val=255.0)
                en_img_psnr = tf.image.psnr(im1, im3, max_val=255.0)
                img_ssim = tf.image.ssim(im1, im2, max_val=255.0)
                en_img_ssim = tf.image.ssim(im1, im3, max_val=255.0)

                print("psnr: ", img_psnr)
                print("en-psnr: ", en_img_psnr)
                print("ssim: ", img_ssim)
                print("en-ssim: ", en_img_ssim)
                tf.print("gloss: ", G_loss_dict)
                tf.print("dloss: ", D_loss_dict)
                tf.print("hloss: ", H_loss_dict)


        # save checkpoint
        checkpoint.save(ep)
