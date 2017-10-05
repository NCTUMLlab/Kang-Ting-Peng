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
    weights = tf.get_variable('weights{}'.format(l_id), initializer=xavier_init([in_dim, out_dim]), trainable=trainable)
    biases  = tf.get_variable('biases{}'.format(l_id), out_dim, initializer=tf.constant_initializer(value), trainable=trainable)    
    return tf.matmul(x, weights) + biases
    
class Model(object):
    def __init__(self, x_dim, z_dim, y_dim):
        self.disc_layer = [x_dim, 1000, 1000, 1000, 1]
        self.gen_layer = [z_dim + y_dim, 1000, 1000, x_dim]
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.lr = 1e-4
        
        def cross_entropy(logit, y):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))
        
        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim], name='y')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')

        
        fake_x = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_real, C_real = self.discriminator(self.x, is_training=True, reuse=False)
        D_fake, C_fake = self.discriminator(fake_x, is_training=False, reuse=True)

        # Cross entropy aux loss
        C_loss = cross_entropy(C_real, self.y) + cross_entropy(C_fake, self.y)   
        
        # GAN D loss
        D_loss = 0.5 * (tf.reduce_mean((D_real - 1)**2) + tf.reduce_mean(D_fake**2))
        self.DC_loss = D_loss + C_loss
        
        # GAN's G loss
        G_loss = 0.5 * tf.reduce_mean((D_fake - 1)**2)
        self.GC_loss = G_loss + C_loss

        # Cosine loss 
        x_norm = tf.norm(self.x, ord=2, axis=1)
        x_unit = self.x / x_norm[:, None]

        x_hat_norm = tf.norm(fake_x, ord=2, axis=1)
        x_hat_unit = fake_x / x_hat_norm[:, None]
        self.cosine_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(x_unit, x_hat_unit), 1))
        
        theta_G = self.get_train_var('Gen')
        theta_D = self.get_train_var('Disc')      
       
        
        self.D_solver = (tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.DC_loss, var_list=theta_D))
        self.G_solver = (tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.GC_loss, var_list=theta_G))
        self.Cos_solver = (tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cosine_loss, var_list=theta_G))
        
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
            for i, (in_dim, out_dim) in enumerate(zip(self.disc_layer, self.disc_layer[1:-1])):
                h = linear_layer(h, in_dim, out_dim, i)
                h = tf.nn.relu(h)
            
            out_gan = linear_layer(h, self.disc_layer[-2], self.disc_layer[-1], 'out_gan')
            out_aux = linear_layer(h, self.disc_layer[-2], self.y_dim, 'out_aux')    
        return out_gan, out_aux 
        
    def get_train_var(self, name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    
    def training_disc(self, sess, x, z, y):
        _, DC_loss = sess.run([self.D_solver, self.DC_loss], feed_dict={self.x: x, self.z: z, self.y: y})
        return DC_loss
    
    def training_gen(self, sess, x, z, y):
        _, GC_loss = sess.run([self.G_solver, self.GC_loss], feed_dict={self.x: x, self.z: z, self.y: y})
        return GC_loss 

    def training_cos(self, sess, x, z, y):
        _, GC_loss = sess.run([self.Cos_solver, self.cosine_loss], feed_dict={self.x: x, self.z: z, self.y: y})
        return GC_loss 
    
    def generate(self, sess, z, y):
        G_sample = sess.run(self.G_sample, feed_dict={ self.z: z, self.y: y})
        return G_sample
        
        
if __name__ == '__main__':
    model = Model(100, 2, 10)
    model.set_model()
    
