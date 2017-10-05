import os
import sys
import numpy as np
import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def linear_layer(x, in_dim, out_dim, l_id, trainable=True):
    value = 0.
    weights = tf.get_variable('weights_{}'.format(l_id), initializer=xavier_init([in_dim, out_dim]), trainable=trainable)
    biases  = tf.get_variable('biases_{}'.format(l_id), out_dim, initializer=tf.constant_initializer(value), trainable=trainable)    
    return tf.matmul(x, weights) + biases

def cross_entropy(logit, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)) 

class Model(object):
    def __init__(self, x_dim, z_dim, y_dim):


        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.disc_layer = [x_dim, 1000, 1000] # No contain dim of output 
        self.gen_layer = [z_dim + y_dim, 1000, 1000, x_dim]
        
        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim], name='y')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')

        
        fake_x = self.generator(self.z, self.y, is_training=True, reuse=False)
        

        # discriminator 
        r_adv, r_aux, r_h = self.discriminator(self.x, is_training=True, reuse=False)
        f_adv, f_aux, f_h = self.discriminator(fake_x, is_training=True, reuse=True)

        
        # Cross entropy aux loss
        self.aux_loss = cross_entropy(r_aux, self.y) + cross_entropy(f_aux, self.y)   
        
        
        # Cosine loss 
        rx_unit = tf.nn.l2_normalize(self.x, dim=1) 
        fx_unit = tf.nn.l2_normalize(fake_x, dim=1)
        
        r_unit = tf.nn.l2_normalize(r_h, dim=1)        
        f_unit = tf.nn.l2_normalize(f_h, dim=1)
        
        self.incosine_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(rx_unit, fx_unit), 1))
        self.pcosine_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(r_unit, f_unit), 1))
        self.ncosine_loss = -self.pcosine_loss 
       
        # GAN's D loss
        self.D_Adv_loss = 0.5 * (tf.reduce_mean((r_adv - 1)**2) + tf.reduce_mean(f_adv**2))
        self.DC_loss = self.D_Adv_loss + self.aux_loss + self.pcosine_loss
        
        
        # GAN's G loss
        self.G_Adv_loss = 0.5 * tf.reduce_mean((f_adv - 1)**2)
        self.GC_loss = self.G_Adv_loss + self.aux_loss
        
        
        # varibles
        theta_Gen = self.get_train_var('Gen') 
        theta_Disc = self.get_train_var('Disc')
        self.theta = [theta_Gen, theta_Disc]
        
        # solver
        self.D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.DC_loss, var_list=theta_Disc)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.GC_loss, var_list=theta_Gen) 
        self.Cos_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.ncosine_loss, var_list=theta_Gen)
        
        
        # -- for using ---------------------
        self.G_sample = self.generator(self.z, self.y, is_training=False, reuse=True)
    
    def generator(self, z, c, is_training, reuse):
        h = tf.concat(axis=1, values=[z, c])
        with tf.variable_scope('Gen', reuse=reuse):
            for i, (in_dim, out_dim) in enumerate(zip(self.gen_layer, self.gen_layer[1:])):
                h = linear_layer(h, in_dim, out_dim, i)
                h = tf.nn.relu(h)
            ret = linear_layer(h, self.gen_layer[-1], self.gen_layer[-1], 'out')
            return ret
     
    def discriminator(self, x, is_training, reuse):
        h = x
        with tf.variable_scope('Disc', reuse=reuse):
            for i, (in_dim, out_dim) in enumerate(zip(self.disc_layer, self.disc_layer[1:])):
                h = linear_layer(h, in_dim, out_dim, i)
                h = tf.nn.tanh(h)
            out_f = h
            out_gan = linear_layer(out_f, self.disc_layer[-1], 1, 'out_gan')
            out_aux = linear_layer(out_f, self.disc_layer[-1], self.y_dim, 'out_aux')    
        return out_gan, out_aux, out_f
    
    def get_train_var(self, name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    
    def training_disc(self, sess, x, z, y):
        _, DC_loss = sess.run([self.D_solver, self.DC_loss], feed_dict={self.x: x, self.z: z, self.y: y})
        return DC_loss
    
    def training_gen(self, sess, x, z, y):
        _, GC_loss = sess.run([self.G_solver, self.GC_loss], feed_dict={self.x: x, self.z: z, self.y: y})
        return GC_loss 

    def training_cos(self, sess, x, z, y):
        _, cos_loss, ori = sess.run([self.Cos_solver, self.pcosine_loss, self.incosine_loss], 
                                    feed_dict={self.x: x, self.z: z, self.y: y})
        return cos_loss, ori
    
    def generate(self, sess, z, y):
        G_sample = sess.run(self.G_sample, feed_dict={ self.z: z, self.y: y})
        return G_sample
    
    def display_var(self):
        for i in self.theta:
            print('-------------------------------------------------------')
            for v in i:
                print v
        
if __name__ == '__main__':
    model = Model(100, 2, 10)
    model.set_model()
    
