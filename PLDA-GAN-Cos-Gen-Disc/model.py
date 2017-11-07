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
        self.latent_dim = 200
        self.enc_layer = [x_dim, 1000, 1000, self.latent_dim]
        self.gen_layer = [z_dim + y_dim, 1000, 1000, x_dim]
        
        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim], name='y')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')

        
        fake_x = self.generator(self.z, self.y, is_training=True, reuse=False)
        
        # Encoding
        r_mu, r_log_sigma = self.encoder(self.x, is_training=True, reuse=False)
        f_mu, f_log_sigma = self.encoder(fake_x, is_training=True, reuse=True)
        
        # latent
        r_eps = tf.random_normal(shape=tf.shape(r_mu))
        r_latent = tf.add(r_mu, tf.multiply(tf.sqrt(tf.exp(r_log_sigma)), r_eps))
        
        f_eps = tf.random_normal(shape=tf.shape(f_mu))
        f_latent = tf.add(f_mu, tf.multiply(tf.sqrt(tf.exp(f_log_sigma)), f_eps))        
        
        # Deconding
        r_x, r_sigma = self.decoder(r_latent, is_training=True, reuse=False)
        f_x, f_Sigma = self.decoder(f_latent, is_training=True, reuse=True)
        

        # discriminator 
        r_adv, r_aux = self.discrminator(r_latent, is_training=True, reuse=False)
        f_adv, f_aux = self.discrminator(f_latent, is_training=True, reuse=True)
        
        # KL loss        
        self.rKL_loss = -0.5 * tf.reduce_sum(1.0 + r_log_sigma - tf.square(r_mu) - tf.exp(r_log_sigma + 1e-10), 1)
        self.rKL_loss = tf.reduce_mean(self.rKL_loss)
        
        self.fKL_loss = -0.5 * tf.reduce_sum(1.0 + f_log_sigma - tf.square(f_mu) - tf.exp(f_log_sigma + 1e-10), 1)
        self.fKL_loss = tf.reduce_mean(self.fKL_loss)
        
        self.KL_loss = self.rKL_loss + self.fKL_loss 
        
        # reconstruction loss 
        self.rRec_loss = 0.5 * tf.reduce_sum(tf.matmul((self.x - r_x), tf.matrix_inverse(r_sigma)) * (self.x - r_x), 1) 
        L = tf.cholesky(r_sigma)
        self.rRec_loss += tf.log(tf.matrix_determinant(L) + 1e-10) 
        self.rRec_loss = tf.reduce_mean(self.rRec_loss)

        self.fRec_loss = 0.5 * tf.reduce_sum(tf.matmul((self.x - r_x), tf.matrix_inverse(r_sigma)) * (self.x - r_x), 1) 
        L = tf.cholesky(r_sigma)
        self.fRec_loss += tf.log(tf.matrix_determinant(L) + 1e-10) 
        self.fRec_loss = tf.reduce_mean(self.fRec_loss)
        
        self.Rec_loss = self.rRec_loss + self.fRec_loss
        
        # Cross entropy aux loss
        self.aux_loss = cross_entropy(r_aux, self.y) + cross_entropy(f_aux, self.y)   
        
  
        # Cosine loss 
        x_norm = tf.norm(r_latent, ord=2, axis=1)
        x_unit = self.x / x_norm[:, None]

        x_hat_norm = tf.norm(f_latent, ord=2, axis=1)
        x_hat_unit = fake_x / x_hat_norm[:, None]
        self.cosine_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(x_unit, x_hat_unit), 1))
        self.gcosine_loss = -self.cosine_loss 
        
        
        # GAN's D loss
        self.D_Adv_loss = 0.5 * (tf.reduce_mean((r_adv - 1)**2) + tf.reduce_mean(f_adv**2))
        self.DC_loss = self.D_Adv_loss + self.aux_loss + self.cosine_loss
        
        
        # GAN's G loss
        self.G_Adv_loss = 0.5 * tf.reduce_mean((f_adv - 1)**2)
        self.GC_loss = self.G_Adv_loss + self.aux_loss


        
        # PLDA loss
        self.Pre_PLDA_loss = self.rKL_loss + self.rRec_loss + self.cosine_loss
        self.PLDA = self.KL_loss + self.Rec_loss
        
        
        # varibles
        self.theta_Gen = self.get_train_var('Gen') 
        self.theta_Enc = self.get_train_var('Enc')
        self.theta_Dec = self.get_train_var('Dec')
        self.theta_Disc = self.get_train_var('Enc')
        self.theta_Disc.extend(self.get_train_var('Disc'))
        

        # solver
        self.Pre_Enc_solver = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.Pre_PLDA_loss, var_list=self.theta_Enc)
        self.Pre_Dec_solver = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.Pre_PLDA_loss, var_list=self.theta_Dec)
        self.Pre_PLDA_solver = tf.group(self.Pre_Enc_solver, self.Pre_Dec_solver)

        self.D_solver = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(self.DC_loss * 10.0, var_list=self.theta_Disc)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.GC_loss, var_list=self.theta_Gen)
        
        self.Enc_solver = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(self.PLDA, var_list=self.theta_Enc)
        self.Dec_solver = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(self.PLDA, var_list=self.theta_Dec)
        self.PLDA_solver = tf.group(self.Enc_solver, self.Dec_solver)
        
        self.Cos_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.gcosine_loss, var_list=self.theta_Gen)
        
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
        
    def encoder(self, x, is_training, reuse):
        h = x
        with tf.variable_scope('Enc', reuse=reuse):
            for i, (in_dim, out_dim) in enumerate(zip(self.enc_layer, self.enc_layer[1:-1])):
                h = linear_layer(h, in_dim, out_dim, i)
                h = tf.nn.tanh(h)
            mu = linear_layer(h, self.enc_layer[-2], self.enc_layer[-1], 'mu')
            log_sigma = linear_layer(h, self.enc_layer[-2], self.enc_layer[-1], 'log_sigma')
        return mu, log_sigma        
        
    def decoder(self, x, is_training, reuse):
        h = x
        with tf.variable_scope('Dec', reuse=reuse):             
            V = tf.get_variable('V', initializer=xavier_init([self.latent_dim, self.x_dim]), trainable=True) 
            '''
            Sigma = tf.get_variable('Sigma', shape=[self.x_dim, self.x_dim], trainable=True,
                                    initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001))
            
            '''
            init = tf.constant(np.load('L.npy'), dtype=tf.float32)
            Sigma = tf.get_variable('Sigma', initializer=init, trainable=True)
            
            M = tf.constant(np.load('mean.npy'), dtype=tf.float32)
            diag = 1e-3 * tf.constant(np.identity(self.x_dim, dtype=np.float32))
            X_Sigma = tf.add(tf.matmul(Sigma, tf.transpose(Sigma)), diag)
            X_mean = tf.add(tf.matmul(h, V), M) 
            
        return X_mean, X_Sigma
     
    def discrminator(self, x, is_training, reuse):
        h = x
        with tf.variable_scope('Disc', reuse=reuse): 
            '''
            h = linear_layer(h, self.latent_dim, self.latent_dim, 'latent1')
            h = tf.nn.relu(h)
            h = linear_layer(h, self.latent_dim, self.latent_dim, 'latent2')
            h = tf.nn.relu(h)
            '''
            out_gan = linear_layer(h, self.latent_dim, 1, 'out_gan')  
            out_aux = linear_layer(h, self.latent_dim, self.y_dim, 'out_aux')   
        return out_gan, out_aux
    
    def get_train_var(self, name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    
    def training_disc(self, sess, x, z, y):
        _, DC_loss = sess.run([self.D_solver, self.DC_loss], feed_dict={self.x: x, self.z: z, self.y: y})
        return DC_loss
    
    def training_gen(self, sess, x, z, y):
        _, GC_loss = sess.run([self.G_solver, self.GC_loss], feed_dict={self.x: x, self.z: z, self.y: y})
        return GC_loss 

    def pre_training_PLDA(self, sess, x, z, y):
        _, KL_loss, Rec_loss = sess.run([self.Pre_PLDA_solver, self.KL_loss, self.Rec_loss], feed_dict={self.x: x, self.z: z, self.y: y})
        return KL_loss, Rec_loss
    
    def training_PLDA(self, sess, x, z, y):
        _, KL_loss, Rec_loss = sess.run([self.PLDA_solver, self.KL_loss, self.Rec_loss], feed_dict={self.x: x, self.z: z, self.y: y})
        return KL_loss, Rec_loss
    
    def training_cos(self, sess, x, z, y):
        _, GC_loss = sess.run([self.Cos_solver, self.cosine_loss], feed_dict={self.x: x, self.z: z, self.y: y})
        return GC_loss 
    
    def generate(self, sess, z, y):
        G_sample = sess.run(self.G_sample, feed_dict={ self.z: z, self.y: y})
        return G_sample
        
        
if __name__ == '__main__':
    model = Model(100, 2, 10)
    model.set_model()
    
