#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import scipy.misc

import tensorflow as tf
import tensorflow.contrib.slim as slim

import my_time

from Tflib import session

import data
import models

import datetime
from functools import partial
import json
import traceback
import os


# # Imlib

# In[2]:





def immerge(images, row, col):
    """Merge images into an image with (row * h) * (col * w).

    `images` is in shape of N * H * W(* C=1 or 3)
    """
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img


# In[3]:



def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """Transform images from [-1.0, 1.0] to [min_value, max_value] of dtype."""
    assert np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5         and (images.dtype == np.float32 or images.dtype == np.float64),         ('The input images should be float64(32) '
         'and in the range of [-1.0, 1.0]!')
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) +
            min_value).astype(dtype)


# In[4]:


def imwrite(image, path):
    """Save an [-1.0, 1.0] image."""
    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return scipy.misc.imsave(path, to_range(image, 0, 255, np.uint8))


# # Pylib

# In[5]:


def mkdir(paths):
    
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


# # Tflib

# In[6]:


def load_checkpoint(ckpt_dir_or_file, session, var_list=None):

    if os.path.isdir(ckpt_dir_or_file):
        ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    restorer = tf.train.Saver(var_list)
    restorer.restore(session, ckpt_dir_or_file)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)


# In[7]:



def summary(tensor_collection,
            summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram'],
            scope=None):
    """Summary.

    usage:
        1. summary(tensor)
        2. summary([tensor_a, tensor_b])
        3. summary({tensor_a: 'a', tensor_b: 'b})
    """
    def _summary(tensor, name, summary_type):
        """Attach a lot of summaries to a Tensor."""
        if name is None:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
            name = re.sub(':', '-', name)

        summaries = []
        if len(tensor.shape) == 0:
            summaries.append(tf.summary.scalar(name, tensor))
        else:
            if 'mean' in summary_type:
                mean = tf.reduce_mean(tensor)
                summaries.append(tf.summary.scalar(name + '/mean', mean))
            if 'stddev' in summary_type:
                mean = tf.reduce_mean(tensor)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                summaries.append(tf.summary.scalar(name + '/stddev', stddev))
            if 'max' in summary_type:
                summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
            if 'min' in summary_type:
                summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
            if 'sparsity' in summary_type:
                summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
            if 'histogram' in summary_type:
                summaries.append(tf.summary.histogram(name, tensor))
        return tf.summary.merge(summaries)

    if not isinstance(tensor_collection, (list, tuple, dict)):
        tensor_collection = [tensor_collection]

    with tf.name_scope(scope, 'summary'):
        summaries = []
        if isinstance(tensor_collection, (list, tuple)):
            for tensor in tensor_collection:
                summaries.append(_summary(tensor, None, summary_type))
        else:
            for tensor, name in tensor_collection.items():
                summaries.append(_summary(tensor, name, summary_type))
        return tf.summary.merge(summaries)


# In[8]:



def counter(start=0, scope=None):
    with tf.variable_scope(scope, 'counter'):
        counter = tf.get_variable(name='counter',
                                  initializer=tf.constant_initializer(start),
                                  shape=(),
                                  dtype=tf.int64)
        update_cnt = tf.assign(counter, tf.add(counter, 1))
        return counter, update_cnt


# In[9]:



def tensors_filter(tensors, filters, combine_type='or'):
    assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
    assert isinstance(filters, (str, list, tuple)), '`filters` should be a string or a list(tuple) of strings!'
    assert combine_type == 'or' or combine_type == 'and', "`combine_type` should be 'or' or 'and'!"

    if isinstance(filters, str):
        filters = [filters]

    f_tens = []
    for ten in tensors:
        if combine_type == 'or':
            for filt in filters:
                if filt in ten.name:
                    f_tens.append(ten)
                    break
        elif combine_type == 'and':
            all_pass = True
            for filt in filters:
                if filt not in ten.name:
                    all_pass = False
                    break
            if all_pass:
                f_tens.append(ten)
    return f_tens


# In[10]:


def trainable_variables(filters=None, combine_type='or'):
    t_var = tf.trainable_variables()
    if filters is None:
        return t_var
    else:
        return tensors_filter(t_var, filters, combine_type)


# In[11]:



# ==============================================================================
# =                                    param                                   =
# ==============================================================================


atts =  ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
n_att = len(atts)
img_size = 128
shortcut_layers = 1
inject_layers = 1
enc_dim = 64
dec_dim = 64
dis_dim = 64
dis_fc_dim = 1024
enc_layers = 5
dec_layers = 5
dis_layers = 5
# training
mode = 'wgan'
epoch = 200
batch_size = 32
lr_base = 0.0002
n_d = 5
b_distribution = 'none'
thres_int = 0.5
test_int = 1
n_sample = 64
# others
use_cropped_img = False
experiment_name = 'training'




# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
sess = session()
tr_data = data.Celeba('./data', atts, img_size, batch_size, part='train', sess=sess, crop=not use_cropped_img)
val_data = data.Celeba('./data', atts, img_size, n_sample, part='val', shuffle=False, sess=sess, crop=not use_cropped_img)

# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers, inject_layers=inject_layers)
D = partial(models.D, n_att=n_att, dim=dis_dim, fc_dim=dis_fc_dim, n_layers=dis_layers)

# inputs
lr = tf.placeholder(dtype=tf.float32, shape=[])

xa = tr_data.batch_op[0]
a = tr_data.batch_op[1]
b = tf.random_shuffle(a)
_a = (tf.to_float(a) * 2 - 1) * thres_int
if b_distribution == 'none':
    _b = (tf.to_float(b) * 2 - 1) * thres_int
elif b_distribution == 'uniform':
    _b = (tf.to_float(b) * 2 - 1) * tf.random_uniform(tf.shape(b)) * (2 * thres_int)
elif b_distribution == 'truncated_normal':
    _b = (tf.to_float(b) * 2 - 1) * (tf.truncated_normal(tf.shape(b)) + 2) / 4.0 * (2 * thres_int)

xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# generate
z = Genc(xa)
xb_ = Gdec(z, _b)
with tf.control_dependencies([xb_]):
    xa_ = Gdec(z, _a)

# discriminate
xa_logit_gan, xa_logit_att = D(xa)
xb__logit_gan, xb__logit_att = D(xb_)

# discriminator losses
if mode == 'wgan':  # wgan-gp
    wd = tf.reduce_mean(xa_logit_gan) - tf.reduce_mean(xb__logit_gan)
    d_loss_gan = -wd
    gp = models.gradient_penalty(D, xa, xb_)
elif mode == 'lsgan':  # lsgan-gp
    xa_gan_loss = tf.losses.mean_squared_error(tf.ones_like(xa_logit_gan), xa_logit_gan)
    xb__gan_loss = tf.losses.mean_squared_error(tf.zeros_like(xb__logit_gan), xb__logit_gan)
    d_loss_gan = xa_gan_loss + xb__gan_loss
    gp = models.gradient_penalty(D, xa)
elif mode == 'dcgan':  # dcgan-gp
    xa_gan_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(xa_logit_gan), xa_logit_gan)
    xb__gan_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(xb__logit_gan), xb__logit_gan)
    d_loss_gan = xa_gan_loss + xb__gan_loss
    gp = models.gradient_penalty(D, xa)

xa_loss_att = tf.losses.sigmoid_cross_entropy(a, xa_logit_att)

d_loss = d_loss_gan + gp * 10.0 + xa_loss_att

# generator losses
if mode == 'wgan':
    xb__loss_gan = -tf.reduce_mean(xb__logit_gan)
elif mode == 'lsgan':
    xb__loss_gan = tf.losses.mean_squared_error(tf.ones_like(xb__logit_gan), xb__logit_gan)
elif mode == 'dcgan':
    xb__loss_gan = tf.losses.sigmoid_cross_entropy(tf.ones_like(xb__logit_gan), xb__logit_gan)

xb__loss_att = tf.losses.sigmoid_cross_entropy(b, xb__logit_att)
xa__loss_rec = tf.losses.absolute_difference(xa, xa_)

g_loss = xb__loss_gan + xb__loss_att * 10.0 + xa__loss_rec * 100.0

#global_step
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

# optim
d_var = trainable_variables('D')
d_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss, var_list=d_var)

g_var = trainable_variables('G')
g_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var,global_step=global_step)

# summary
d_summary = summary({
    d_loss_gan: 'd_loss_gan',
    gp: 'gp',
    xa_loss_att: 'xa_loss_att',
}, scope='D')

g_summary = summary({
    xb__loss_gan: 'xb__loss_gan',
    xb__loss_att: 'xb__loss_att',
    xa__loss_rec: 'xa__loss_rec',
}, scope='G')

lr_summary = summary({lr: 'lr'}, scope='Learning_Rate')

d_summary = tf.summary.merge([d_summary, lr_summary])

# sample
x_sample = Gdec(Genc(xa_sample, is_training=False), _b_sample, is_training=False)


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# iteration counter
it_cnt, update_cnt = counter()

# saver
saver = tf.train.Saver(max_to_keep=1)

# summary writer
summary_writer = tf.summary.FileWriter('./output/summaries', sess.graph)

# initialization
ckpt_dir = './output/%s/'%experiment_name

try:
    load_checkpoint(ckpt_dir, sess)
except:
    sess.run(tf.global_variables_initializer())

# train
try:
    # data for sampling
    xa_sample_ipt, a_sample_ipt = val_data.get_next()
    b_sample_ipt_list = [a_sample_ipt]  # the first is for reconstruction
    for i in range(len(atts)):
        tmp = np.array(a_sample_ipt, copy=True)
        tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
        tmp = data.Celeba.check_attribute_conflict(tmp, atts[i], atts)
        b_sample_ipt_list.append(tmp)

    it_per_epoch = len(tr_data) // (batch_size * (n_d + 1))
    max_it = epoch * it_per_epoch
    for it in range(sess.run(it_cnt), max_it):
        with my_time.Timer(is_output=False) as t:
            sess.run(update_cnt)

            # which epoch
            epoch = it//it_per_epoch
            it_in_epoch = it%it_per_epoch + 1

            # learning rate
            lr_ipt = lr_base / (10 ** (epoch // 100))

            # train D
            for i in range(n_d):
                d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={lr: lr_ipt})
            summary_writer.add_summary(d_summary_opt, it)

            # train G
            g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={lr: lr_ipt})
            summary_writer.add_summary(g_summary_opt, it)
            
            i_global = sess.run(global_step)
            
            # display
            if (it + 1) % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d) Time: %s! ,global_step :(%d)" % (epoch, it_in_epoch, it_per_epoch, t,i_global))

                    
            

            # save
            if (it + 1) % 300 == 0:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch),global_step=global_step)
                print('Model is saved at %s!' % save_path)
            
                    
        # sample
            if (it + 1) % 100 == 0:
                x_sample_opt_list = [xa_sample_ipt, np.full((n_sample, img_size, img_size // 10, 3), -1.0)]
                for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                    _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
                    if i > 0:   # i == 0 is for reconstruction
                        _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int / thres_int
                    x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
                sample = np.concatenate(x_sample_opt_list, 2)

                save_dir = './output/%s/sample_training' % experiment_name
                mkdir(save_dir)
                imwrite(immerge(sample, n_sample, 1), '%s/Epoch_(%d)_(%dof%d)(g:%d).jpg' % (save_dir, epoch, it_in_epoch, it_per_epoch,i_global))
except:
    traceback.print_exc()
finally:
    save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch),global_step=global_step)
    print('Model is saved at %s!' % save_path)
    sess.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




