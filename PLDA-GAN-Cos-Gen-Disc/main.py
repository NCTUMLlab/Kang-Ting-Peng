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
        
def display_var(theta_var):
    print('-------------------------------------------------------')
    for v in theta_var:
        print v
    

if __name__ == '__main__':

    dt = Dataset('/home/kt/NIST_npy/', one_hot=True)
    mb_size = 100
    x_dim = dt.train_data.shape[1]
    y_dim = dt.train_label.shape[1]
    z_dim = 100
    d_steps = 2
    iteration = 40000   
    path = 'save/'
    check_path(path)
    
    model = Model(x_dim, z_dim, y_dim)
    theta_Gen = model.theta_Gen
    theta_Enc = model.theta_Enc
    theta_Dec = model.theta_Dec
    theta_Disc = model.theta_Disc
    
    display_var(theta_Gen)
    display_var(theta_Enc)
    display_var(theta_Dec)
    display_var(theta_Disc)
    #raw_input('enter')

    record_D_loss = []
    record_G_loss = []
    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        for it in range(1000):
            x_mb, y_mb = dt.next_batch(mb_size)
            z_mb = sample_z(mb_size, z_dim)
            
            KL_loss_curr = 0.
            Rec_loss_curr = 0.
            KL_loss_curr, Rec_loss_curr = model.pre_training_PLDA(sess, x_mb, z_mb, y_mb)

            if it % 100 == 0:
                print('------------------------ PreIter {} ------------------------'.format(it))
                print('KL_loss: {:.6}; Rec_loss: {:.6}'.format(KL_loss_curr, Rec_loss_curr))
                
        for it in range(iteration):
            x_mb, y_mb = dt.next_batch(mb_size)
            z_mb = sample_z(mb_size, z_dim)   
            KL_loss_curr = 0.
            Rec_loss_curr = 0.
            DC_loss_curr = 0.
            GC_loss_curr = 0.
            Cos_loss_curr = 0.
            KL_loss_curr, Rec_loss_curr = model.training_PLDA(sess, x_mb, z_mb, y_mb)
            model.training_disc(sess, x_mb, z_mb, y_mb)
            model.training_disc(sess, x_mb, z_mb, y_mb)
            DC_loss_curr = model.training_disc(sess, x_mb, z_mb, y_mb)
            GC_loss_curr = model.training_gen(sess, x_mb, z_mb, y_mb)
            Cos_loss_curr = model.training_cos(sess, x_mb, z_mb, y_mb)
            if it % 100 == 0:
                record_D_loss.append(DC_loss_curr)
                record_G_loss.append(GC_loss_curr)
                saver.save(sess, 'save/model.ckpt')
                print('------------------------ Iter {} ------------------------'.format(it))
                print('KL_loss: {:.6}; Rec_loss: {:.6}'.format(KL_loss_curr, Rec_loss_curr))
                print('D_loss: {:.4}; G_loss: {:.4}; Cos_loss: {:.4}'.format(DC_loss_curr, GC_loss_curr, Cos_loss_curr))
                
        saver.save(sess, path + 'model.ckpt')
        np.save(path + 'record_D_loss.npy', record_D_loss)
        np.save(path + 'record_G_loss.npy', record_G_loss)         
