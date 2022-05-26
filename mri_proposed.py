import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='brain')
py.arg('--datasets_dir', default='/home/Alexandrite/smin/cycle_git/')
py.arg('--load_size', type=int, default=256)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay', type=int, default=25)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
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


######################## ours brain random sampling ############################


clean = np.array(data.image_read(py.join('/home/Alexandrite/smin/cycle_git/input/fastmri/ours/brain/brain', 'clean_R4_8')), dtype=np.float32)
noisy = np.array(data.image_read(py.join('/home/Alexandrite/smin/cycle_git/input/fastmri/ours/brain/brain', 'noisy_R4_8')), dtype=np.float32)
mask = np.array(data.image_read(py.join('/home/Alexandrite/smin/cycle_git/input/fastmri/ours/brain/brain', 'mask_R4_8')), dtype=np.float32)


################################################################################

"""
A_img_paths = py.glob(py.join(args.datasets_dir, 'brain', 'half_clean'), '*.png')
B_img_paths = py.glob(py.join(args.datasets_dir, 'brain', 'half_noisy'), '*.png')
"""

"""
A_img_paths = py.glob(py.join(args.datasets_dir, 'knees', 'train_clean_20'), '*.png')
B_img_paths = py.glob(py.join(args.datasets_dir, 'knees', 'train_noisy_20'), '*.png')
"""

"""
A_img_paths = py.glob(py.join('/home/Alexandrite/smin/FastMRI/ours/brain', 'clean_R4_8'), '*.png')
B_img_paths = py.glob(py.join('/home/Alexandrite/smin/FastMRI/ours/brain', 'noisy_R4_8'), '*.png')
m_img_paths = py.glob(py.join('/home/Alexandrite/smin/FastMRI/ours/brain', 'mask_R4_8'), '*png')

A_img_paths = A_img_paths[:50]
B_img_paths = B_img_paths[50:]
"""

clean = np.array(data.image_read(py.join('/home/Alexandrite/smin/cycle_git/input/fastmri/ours/brain/brain', 'clean_R4_8')), dtype=np.float32)
noisy = np.array(data.image_read(py.join('/home/Alexandrite/smin/cycle_git/input/fastmri/ours/brain/brain', 'noisy_R4_8')), dtype=np.float32)
mask = np.array(data.image_read(py.join('/home/Alexandrite/smin/cycle_git/input/fastmri/ours/brain/brain', 'mask_R4_8')), dtype=np.float32)

"""
clean = np.array(data.image_read(py.join('/home/Alexandrite/smin/cycle_git/data/brain', 'db_train')), dtype=np.float32)
noisy = np.array(data.image_read(py.join('/home/Alexandrite/smin/cycle_git/data/brain', 'train_noisy')), dtype=np.float32)
mask_1 = np.array(data.image_read(py.join('/home/Alexandrite/smin/cycle_git/data/mask/radial/mask_1')), dtype=np.float32)
mask = mask_1
for i in range(99):
    mask = np.concatenate([mask, mask_1], axis=0)"""

#m_img_paths = py.glob(py.join('/home/Alexandrite/smin/FastMRI/ours/brain', 'mask_R4_8'), '*png')


clean = clean / 255.0
noisy = noisy / 255.0
clean = clean * 2 -1
noisy = noisy * 2 -1
mask = mask /255.0


clean = clean[:50]
a_mask = mask[:50]
noisy = noisy[50:]
b_mask = mask[50:]

clean = data.image_division(data.image_augmentation(clean), patch_size=(256, 256))
noisy = data.image_division(data.image_augmentation(noisy), patch_size=(256, 256))
a_mask = data.image_division(data.image_augmentation(a_mask), patch_size=(256, 256))
b_mask = data.image_division(data.image_augmentation(b_mask), patch_size=(256, 256))

len_dataset = len(clean)
A_B_dataset = tf.data.Dataset.from_tensor_slices((clean, noisy, a_mask, b_mask))  # If you don't have enough memory, you can use tf.data.Dataset.from_generator
A_B_dataset = A_B_dataset.cache().shuffle(len(clean), reshuffle_each_iteration=True).batch(1).prefetch(tf.data.experimental.AUTOTUNE)


#A_img_paths = np.concatenate([A_img_paths, A_img_paths], axis=0)
#B_img_paths = np.concatenate([B_img_paths, B_img_paths], axis=0)

#mask_zip = zip(a_mask, b_mask)


"""
A_img_paths = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee', 'singlecoil_val', 'clean_R4_8'), '*.png')
B_img_paths = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee', 'singlecoil_val', 'noisy_R4_8'), '*.png')
m_img_paths = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee', 'singlecoil_val', 'mask_R4_8'), '*png')

A_img_paths = A_img_paths[:100]
A_mask = m_img_paths[:100]
B_img_paths = B_img_paths[100:200]
B_mask = m_img_paths[100:200]
mask_zip = zip(A_mask, B_mask)
"""

#A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, training=True, repeat=False)
#A_B_dataset = A_B_dataset.cache().repeat().shuffle(len(len_dataset), reshuffle_each_iteration=True).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

#A2B_pool = data.ItemPool(args.pool_size)
#B2A_pool = data.ItemPool(args.pool_size)

#A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'db_valid'), '*.png')
#B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'noisy'), '*.png')

"""
A_img_paths_test = py.glob(py.join(args.datasets_dir, 'brain', 'db_valid'), '*.png')
B_img_paths_test = py.glob(py.join(args.datasets_dir, 'brain', 'noisy'), '*.png')"""
#B_img_paths_test.sort()

"""
A_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee',  'singlecoil_val', 'clean_R4_8'), '*.png')
B_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee',  'singlecoil_val', 'noisy_R4_8'), '*.png')

A_img_paths_test = A_img_paths_test[-100:]
B_img_paths_test = B_img_paths_test[-100:]
"""


A_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/cycle_git/input/fastmri/ours/brain/brain', 'clean_R4_8_val'), '*.png')
B_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/cycle_git/input/fastmri/ours/brain/brain', 'noisy_R4_8_val'), '*.png')

#mask = np.array(image_read("/home/Alexandrite/smin/ISCL_MRI/data/mask/radial/mask_1/mask_10.png"), dtype=np.float32)

"""
A_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee', 'singlecoil_val','clean_R6_6'), '*.png')
B_img_paths_test = py.glob(py.join('/home/Alexandrite/smin/FastMRI', 'knee', 'singlecoil_val','noisy_R6_6'), '*.png')
A_img_paths_test = A_img_paths_test[-500:]
B_img_paths_test = B_img_paths_test[-500:]
"""

A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size, args.crop_size, shuffle=False, training=False, repeat=True)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================s

G = module.Gen_with_adain(n_blocks=9)

D_A = module.ConvDiscriminator()
D_B = module.ConvDiscriminator()
#D_A = module.Discriminator()
#D_B = module.Discriminator()

H = module.Extractor()

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()
mae_loss_fn = tf.losses.MeanAbsoluteError()


"""
G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
H_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)
H_optimizer = keras.optimizers.Adam(learning_rate=H_lr_scheduler, beta_1=args.beta_1)
"""

G_opt = tfa.optimizers.RectifiedAdam(learning_rate=args.lr, beta_1=0.9, warmup_proportion=0.0,total_steps=int(len_dataset * args.epochs), min_lr=1e-7)
D_opt = tfa.optimizers.RectifiedAdam(learning_rate=args.lr, beta_1=0.9, warmup_proportion=0.0,total_steps=int(len_dataset * args.epochs), min_lr=1e-7)
H_opt = tfa.optimizers.RectifiedAdam(learning_rate=args.lr, beta_1=0.9, warmup_proportion=0.0,total_steps=int(len_dataset * args.epochs), min_lr=1e-7)
G_optimizer = tfa.optimizers.SWA(G_opt, start_averaging=int(len_dataset * args.epochs * 0.6),average_period=1)
D_optimizer = tfa.optimizers.Lookahead(D_opt, sync_period=6, slow_step_size=0.5)
H_optimizer = tfa.optimizers.Lookahead(H_opt, sync_period=6, slow_step_size=0.5)

#H_opt = tfa.optimizers.RectifiedAdam(learning_rate=args.lr*10, beta_1=0.9, warmup_proportion=0.0, total_steps=int(len_dataset*args.epochs), min_lr=1e-7)
#H_optimizer = tfa.optimizers.Lookahead(H_opt, sync_period=6, slow_step_size=0.5)



#pre_output_dir = py.join(args.datasets_dir, 'pre_output', args.output_date, args.dir_num)
pre_output_dir = py.join(args.datasets_dir, 'pre_output', '0331', '6')
#py.mkdir(output_dir)


############## 만약 pre training 이 H 학습시킨거면 H추가해야함#####################################################
#tl.Checkpoint(dict(G=G, H=H), py.join(pre_output_dir, 'checkpoints')).restore()



# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

#tf.random.set_seed(5)
z = tf.random.normal([1,64], 0, 1, dtype=tf.float32)
#z = tf.ones([1,64])

@tf.function
def train_G(A, B, a_mask, b_mask, h_w):
    with tf.GradientTape() as t:
        z1 = z+tf.random.normal(tf.shape(z), mean=0.0, stddev=1.0, dtype=tf.float32)*1e-1
        A2B = G(A, z=z1,training=True)
        B2A = G(B, training=True)
        #############################  B-B2A <-> H(B)

        A2B2A = G(A2B, training=True)
        B2A2B = G(B2A, z=z1, training=True)
        #A2A = G(A,  training=True)
        #B2B = G(B, z=z1,training=True)

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

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

        #ft_A, kspace_a = gan.sampling_matrix(A, a_mask)
        #ft_A2B, kspace_a2 = gan.sampling_matrix(A2B, a_mask)
        ft_B, kspace_b = gan.sampling_matrix(B, b_mask)
        ft_B2A, kspace_b2 = gan.sampling_matrix(B2A, b_mask)

        bypass_loss = tf.reduce_mean(tf.abs(B2A - clean_H)) + tf.reduce_mean(tf.abs(A - y_hat_j))

        sampling_loss = tf.reduce_mean(tf.abs(ft_B - ft_B2A)) #+ tf.reduce_mean(tf.abs(ft_A - ft_A2B))

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + bypass_loss*h_w # + 0.1*sampling_loss

    G_grad = t.gradient(G_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))

    return ft_B, kspace_b, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'bypass_loss': bypass_loss,
                      'sampling_loss': sampling_loss
                      }


@tf.function
def train_D(A, B):
    with tf.GradientTape() as t:

        z1 = z + tf.random.normal(tf.shape(z), mean=0.0, stddev=1.0, dtype=tf.float32) * 1e-1
        A2B = G(A, z=z1, training=False)
        B2A = G(B, training=False)

        A_d_logits = D_A(A, training=True)
        B2A_d_logits= D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        #D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
        #D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        ######## bst loss
        clean_H = B - H(B, training=False)
        noisy_H = A + H(B, training=False)
        fake_clean = D_A(clean_H, training=True)
        fake_noisy = D_B(noisy_H, training=True)

        bst_loss = tf.reduce_mean(tf.math.square(fake_noisy)) + tf.reduce_mean(tf.math.square(fake_clean))

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss)+ bst_loss

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'bst_loss': bst_loss
            }


@tf.function
def train_H(A,B, b_mask):
    with tf.GradientTape() as t:
        z1 = z+tf.random.normal(tf.shape(z), mean=0.0, stddev=1.0, dtype=tf.float32)*1e-1
        #z1 = z
        #z1 = tf.random.normal([1, 64], 0, 1, dtype=tf.float32)

        n_hat_i = H(B, training=True) # A가 noisy B가 clean noise
        n_bar_i = B - G(B, training=True) # nosiy - clean noise
        x_hat_j = G(A,  z=z1, training=True) # fake noisy
        n_tilda_j = H(x_hat_j, training=True) # fake noisy noise

        pseudo_loss = tf.reduce_mean(tf.abs(n_hat_i - n_bar_i))
        noise_consistency = tf.reduce_mean(tf.abs(x_hat_j - A - n_tilda_j))

        ft_B, k = gan.sampling_matrix(B, b_mask)
        ft_B2A, _ = gan.sampling_matrix(B-n_hat_i, b_mask)

        sampling_loss = tf.reduce_mean(tf.abs(ft_B - ft_B2A))

        loss = pseudo_loss + noise_consistency # + 0.1*sampling_loss

    H_grad = t.gradient(loss, H.trainable_variables)
    H_optimizer.apply_gradients(zip(H_grad, H.trainable_variables))

    return {'pseudo_loss': pseudo_loss,
            'noise_consistency': noise_consistency,
            'sample': sampling_loss}


def train_step(A, B, a_mask, b_mask, curr_epoch):


    if curr_epoch < args.epoch_decay:
        a, b, G_loss_dict = train_G(A, B,a_mask, b_mask, 10)
    else:
        a, b, G_loss_dict = train_G(A, B, a_mask,b_mask, 10)
    #a, b, G_loss_dict = train_G(A, B, a_mask, b_mask)

    # cannot autograph `A2B_pool`
    #A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    #B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B)

    #H_loss_dict = train_H(A, B)
    #H_loss_dict = train_H(A, B)
    H_loss_dict = train_H(A, B, b_mask)



    a= np.log(np.abs(a) + 1e-9)
    b= np.log(np.abs(b) + 1e-9)

    #a = (a - np.min(a)) / (np.max(a) - np.min(a))
    #b = (b - np.min(b)) / (np.max(b) - np.min(b))
    #b *= 255
    #b *= 255

    #print(np.max(ft_A), np.min(ft_A2B2A))
    #temp = np.array((ft_A[0]+1)*0.5*255).astype(np.uint8)
    #im = Image.fromarray(temp)
    #im.save('/home/Alexandrite/smin/cycle_git/res/sampling_matrix_2.tif')
    """
    temp = np.array(a)
    im = Image.fromarray(temp)
    im.save('/home/Alexandrite/smin/cycle_git/res/sampling_matrix_22.tif')
    temp = np.array(b)
    im = Image.fromarray(temp)
    im.save('/home/Alexandrite/smin/cycle_git/res/sampling_matrix_23.tif')
    """
    return G_loss_dict, D_loss_dict, H_loss_dict

def train_step_D(A, B):
    D_loss_dict = train_D(A, B)
    return D_loss_dict


@tf.function
def sample(A, B):
    z = tf.random.normal([1, 64], 0, 1, dtype=tf.float32)

    A2B = tf.clip_by_value(G(A, z=z, training=False), -1.0, 1.0)
    B2A = tf.clip_by_value(G(B, training=False), -1.0, 1.0)
    H2A = B - H(B, training=False)
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

####data loader
"""
class LoaderWrapper:
    def __init__(self, dataloader, n_step):
        self.step = n_step
        self.idx = 0
        self.iter_loader = iter(dataloader)

    def __iter__(self):
        return self

    def __len__(self):
        return self.step

    def __next__(self):
        # if reached number of steps desired, stop
        if self.idx == self.step:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1
        # while True
        try:
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)
"""


############



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
        # train for an epoch
        #mask_zip = zip(a_mask, b_mask)

        for A, B , a_mask, b_mask in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):

            #img = Image.open(a_mask)
            #a_mask = np.array(img, dtype=np.float32)
            #a_mask /= 255
            #a_mask = tf.convert_to_tensor(a_mask)
            #img = Image.open(b_mask)
            #b_mask = np.array(img, dtype=np.float32)
            #b_mask /= 255
            #b_mask = tf.convert_to_tensor(b_mask)


            G_loss_dict, D_loss_dict, H_loss_dict = train_step(A, B, a_mask, b_mask, ep)
            #train_step_D(A,B)
            #train_step_D(A, B)
            #train_H(A,B)
            #train_H(A,B)
            #train_H(A,B)
            #train_H(A,B)


            # # summary
            """
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=D_optimizer.iterations, name='D_losses')
            tl.summary(H_loss_dict, step=H_optimizer.iterations, name='H_losses')
            tl.summary({'learning rate': G_opt.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')
            """
            # sample
            if G_optimizer.iterations.numpy() % 400 == 0:
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
                #image0_ph = tf.placeholder(tf.float32)
                #image1_ph = tf.placeholder(tf.float32)

                #img_lpips = lpips_tf.lpips(im1, im2, model='net-lin', net='alex')
                #en_lpips = lpips_tf.lpips(im1, im3, model='net-lin', net='alex')

                print("psnr: ", img_psnr)
                print("en-psnr: ", en_img_psnr)
                print("ssim: ", img_ssim)
                print("en-ssim: ", en_img_ssim)
                #print("lpips: ", img_lpips)
                #print("en-lpips: ", en_lpips)
                tf.print("gloss: ", G_loss_dict)
                tf.print("dloss: ", D_loss_dict)
                tf.print("hloss: ", H_loss_dict)


        # save checkpoint
        checkpoint.save(ep)
