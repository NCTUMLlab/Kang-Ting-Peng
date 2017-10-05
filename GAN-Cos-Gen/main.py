import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import Dataset
from model import Model

def sample_z(m, n):
    return np.random.normal(loc=0., scale=1., size=[m, n])
    #return np.random.uniform(-5., 5., size=[m, n])

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':

    dt = Dataset('../NIST_npy/', one_hot=True)
    mb_size = 100
    x_dim = dt.train_data.shape[1]
    y_dim = dt.train_label.shape[1]
    z_dim = 100
    iteration = 10000   
    path = 'save/'
    check_path(path)
    
    model = Model(x_dim, z_dim, y_dim)
    record_D_loss = []
    record_G_loss = []
    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        for it in range(iteration):
            x_mb, y_mb = dt.next_batch(mb_size)
            z_mb = sample_z(mb_size, z_dim)

            model.training_disc(sess, x_mb, z_mb, y_mb)
            model.training_disc(sess, x_mb, z_mb, y_mb)
            DC_loss_curr = model.training_disc(sess, x_mb, z_mb, y_mb)
            GC_loss_curr = model.training_gen(sess, x_mb, z_mb, y_mb)
            Cos_loss_curr = model.training_cos(sess, x_mb, z_mb, y_mb)
            if it % 100 == 0:
                record_D_loss.append(DC_loss_curr)
                record_G_loss.append(GC_loss_curr)
                saver.save(sess, 'save/model.ckpt')
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; Cos_loss: {:.4}'.format(it, DC_loss_curr, GC_loss_curr, Cos_loss_curr))
                
        saver.save(sess, path + 'model.ckpt')
        np.save(path + 'record_D_loss.npy', record_D_loss)
        np.save(path + 'record_G_loss.npy', record_G_loss)         
