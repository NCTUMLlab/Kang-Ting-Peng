import os
import sys
import numpy as np
import collections

class Dataset(object):
    def __init__(self, file_path, one_hot=False):
        self._train_data = np.load(file_path + 'train_data.npy').astype(np.float32)
        self._train_label = np.load(file_path + 'train_label.npy').astype(np.int32)
        self._test_data = np.load(file_path + 'test_data.npy').astype(np.float32)
        self._verify_data = np.load(file_path + 'verify_data.npy').astype(np.float32)
        
        self._target_status = np.load(file_path + 'target_status.npy')
        self._train_duration = np.load(file_path + 'train_duration.npy')
        self._test_duration = np.load(file_path + 'test_duration.npy')
        self._trial_set = np.load(file_path + 'trial_set.npy')
        
        self._ori_train_data = self._train_data
        self._ori_train_label = self._train_label 
        self._ori_test_data = self._test_data 
        self._ori_verify_data = self._verify_data
        self._bincout = np.bincount(self._ori_train_label)
        self._one_hot = one_hot
        
        if np.shape(self._train_data)[0] != np.shape(self._train_label)[0] :
            raise ValueError('number of x is not equal to number of y')
            
        self._n_examples = np.shape(self._train_data)[0]
        self._n_class = np.max(self._train_label) + 1
        
        if one_hot == True:
            labels = np.zeros([self._n_examples, self._n_class], dtype=np.int32)
            labels[np.arange(self._n_examples), self._train_label] = 1
            self._train_label = labels
        
        self._ln = False
        self._dict = False
        self._epochs_completed = 0 
        self._index_in_epoch = 0
        
    @property
    def bincout(self):
        return self._bincout   
    
    @property
    def n_examples(self):
        return self._n_examples

    @property
    def n_class(self):
        return self._n_class
    
    @property
    def train_data(self):
        return self._train_data
    
    @property
    def train_label(self):
        return self._train_label
    
    @property
    def train_duration(self):
        return self._train_duration  
    
    @property
    def test_data(self):
        return self._test_data

    @property
    def test_duration(self):
        return self._test_duration
    
    @property
    def verify_data(self):
        return self._verify_data
    
    @property
    def target_status(self):
        return self._target_status  
    
    @property
    def trial_set(self):
        return self._trial_set  
    
    # original data
    @property
    def ori_train_data(self):
        return self._ori_train_data
    
    @property
    def ori_train_label(self):
        return self._ori_train_label
    
    @property
    def ori_test_data(self):
        return self._ori_test_data
    
    @property
    def ori_verify_data(self):
        return self._ori_verify_data
    
    @property
    def train_dict(self):
        if self._dict == False:
            raise ValueError('Did not do the dict') 
        else:
            return self._train_dict
    
    @property
    def matW(self):
        if self._ln == False:
            raise ValueError('Did not do the lengthnorm')  
        else:
            return self._W 
        
    def sort(self):
        """Sort speaker ivector dependent on labels"""
        if self._one_hot == True:
            tmp_label = np.argmax(self._train_label, axis=1)
        else:
            tmp_label = self._train_label
        
        idx = np.argsort(tmp_label)
        self._train_label = self._train_label[idx]
        self._train_data = self._train_data[idx]
        print('sorting')
        
    def speaker_dict(self):
        """construct speaker dictionary data with same speaker """
        train_dict = collections.defaultdict(list)
        if self._one_hot == True:
            tmp_label = np.argmax(self._train_label, axis=1)
        else:
            tmp_label = self._train_label
            
        for d, l in zip(self._train_data, tmp_label):
            train_dict[l].append(d)  
        self._train_dict = [np.vstack(train_dict[k]) for k in range(self._n_class)]
        self._dict = True
        print('Creating dict')
             
    def special_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if self._dict == False:
            raise ValueError('Did not do the dict')  
        start = self._index_in_epoch 
        if start + batch_size > self._n_examples:
            # Finished epoch
            self._epochs_completed += 1
            
            # Get the rest examples in this epoch
            rest_n_examples = self._n_examples - start
            x_rest_part = self._train_data[start: self._n_examples]
            y_rest_part = self._train_label[start: self._n_examples]
            
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._n_class)
                np.random.shuffle(perm)
                self._train_data = np.vstack([self._train_dict[k] for k in perm])
                self._train_label = np.asarray([k for k in perm for l in range(len(self._train_dict[k]))], dtype=np.int32)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_n_examples
            end = self._index_in_epoch
            x_new_part = self._train_data[start:end]
            y_new_part = self._train_label[start:end]
            
            ret_x = np.concatenate((x_rest_part, x_new_part), axis=0)
            
            if self._one_hot == True:
                ret_y = np.concatenate((y_rest_part, y_new_part), axis=0)
            else:  
                ret_y = np.hstack((y_rest_part, y_new_part))
                
            return ret_x, ret_y 
        
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._train_data[start:end], self._train_label[start:end]
        
    def next_batch(self, batch_size, shuffle=True):
        
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._n_examples)
            np.random.shuffle(perm0)
            self._train_data = self._train_data[perm0]
            self._train_label = self._train_label[perm0]
            
        if start + batch_size > self._n_examples:
            # Finished epoch
            self._epochs_completed += 1
            
            # Get the rest examples in this epoch
            rest_n_examples = self._n_examples - start
            x_rest_part = self._train_data[start: self._n_examples]
            y_rest_part = self._train_label[start: self._n_examples]
            
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._n_examples)
                np.random.shuffle(perm)
                self._train_data = self._train_data[perm]
                self._train_label = self._train_label[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_n_examples
            end = self._index_in_epoch
            x_new_part = self._train_data[start:end]
            y_new_part = self._train_label[start:end]
            
            ret_x = np.concatenate((x_rest_part, x_new_part), axis=0)
            
            if self._one_hot == True:
                ret_y = np.concatenate((y_rest_part, y_new_part), axis=0)
            else:  
                ret_y = np.hstack((y_rest_part, y_new_part))
                
            return ret_x, ret_y 
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._train_data[start:end], self._train_label[start:end]
        
    def length_norm(self):
        print('length normalize')
        # train data 
        self._ln = True
        self._mean_ivec = np.mean(self._train_data, axis=0)
        minus_ivec = np.subtract(self._train_data, self._mean_ivec)
        ivec_norm = np.linalg.norm(minus_ivec, ord=2, axis=1)
        data = (minus_ivec.T  / ivec_norm).T
        cov_data = np.cov(data.T)
        U, D, V = np.linalg.svd(cov_data)
        W = V.T * np.diag(1. / (np.sqrt(np.diag(D)) + 1e-10))
        self._train_data = np.dot(data, W).astype(np.float32)
        self._W  = W
        
        # test data
        minus_ivec = np.subtract(self._test_data, self._mean_ivec)
        ivec_norm = np.linalg.norm(minus_ivec, ord=2, axis=1)
        data = (minus_ivec.T  / ivec_norm).T
        self._test_data = np.dot(data, self._W).astype(np.float32)
        
        # verify data
        minus_ivec = np.subtract(self._verify_data, self._mean_ivec)
        ivec_ver_norm = np.linalg.norm(minus_ivec, ord=2, axis=1)
        data = (minus_ivec.T / ivec_ver_norm).T
        self._verify_data = np.dot(data, self._W).astype(np.float32)
        
    def add_training_data(self, x, y):
        print('Adding training data')
        if np.shape(self._train_data)[0] != np.shape(self._train_label)[0] :
            raise ValueError('number of x is not equal to number of y')  
        
        if self._one_hot == True:
            if len(y.shape) == 2 and np.shape(self._train_label)[1] == self._n_class:
                self._train_label = np.vstack([self._train_label, y]) 
            else:
                raise ValueError('shape of y is not correct')  
        else:
            if len(y.shape) == 1 and np.max(y) < self._n_class:
                self._train_label = np.hstack([self._train_label, y]) 
            else:
                raise ValueError('shape of y is not correct')  
                
        if np.shape(x)[1] == np.shape(self._train_data)[1]:
            self._train_data = np.vstack([self._train_data, x])
        else:
            raise ValueError('shape of x is not correct') 
            
        self._n_examples = np.shape(self._train_data)[0]
            
    def compute_eer(self, scores):
        scores = np.reshape(scores,[-1, 1])
        assert scores.shape[0] == self._target_status.shape[0], ('scores.shape: %s target_status.shape: %s' % 
                                                                 (scores.shape, self._target_status))

        x = self._target_status[np.argsort(scores.T)]
        x_1 = 0 + x
        x_0 = 1 - x


        FN = np.true_divide(np.cumsum(x_1), (np.sum(x_1) + 1e-10))
        TN = np.true_divide(np.cumsum(x_0), (np.sum(x_0) + 1e-10))
        FP = np.subtract(1., TN)
        TP = np.subtract(1., FN)
     
        FNR = np.true_divide(FN, np.add(np.add(TP, FN), 1e-10))
        FPR = np.true_divide(FP, np.add(np.add(TN, FP), 1e-10))
        difs = np.subtract(FNR, FPR)

        idx1 = np.where(difs < 0)[0][-1]
        idx2 = np.where(difs >= 0)[0][0]

        x = [FNR[idx1], FPR[idx1]]
        y = [FNR[idx2], FPR[idx2]]
        a = np.true_divide((x[0] - x[1]), (y[1] - x[1] - y[0] + x[0]))
        eer = 100. * (x[0] + a * (y[0] - x[0]))
        return eer 
        
    def compute_dcf(self, scores):
        scores = np.reshape(scores, [-1, 1])
        assert scores.shape[0] == self._target_status.shape[0], ('scores.shape: %s target_status.shape: %s' %
                                                                 (scores.shape, self._target_status))
    
        assert scores.shape[0] == self._trial_set.shape[0], ('scores.shape: %s trial_set: %s' % 
                                                             (scores.shape, self._trial_set.shape))


        target_status_prog = self._target_status[self._trial_set == 0]
        target_status_eval = self._target_status[self._trial_set == 1]
        scores_prog = scores[self._trial_set == 0] 
        scores_eval = scores[self._trial_set == 1]
        
        Status = target_status_prog[scores_prog.argsort()]
        Status_1 = Status
        Status_0 = 1 - Status
        M = np.true_divide(np.cumsum(Status_1), np.sum(Status_1))
        F = np.subtract(1., np.true_divide(np.cumsum(Status_0), np.sum(Status_0)))

    
        dcf14_prog = np.amin(np.add(M, np.multiply(100., F)))
        Status = target_status_eval[scores_eval.argsort()]

        Status_1 = Status
        Status_0 = 1 - Status

        M = np.true_divide(np.cumsum(Status_1), np.sum(Status_1))
        F =  np.subtract(1., np.true_divide(np.cumsum(Status_0), np.sum(Status_0)))
        dcf14_eval = np.amin(np.add(M, np.multiply(100, F)))
        return dcf14_prog, dcf14_eval
    
    def PLDA_score(self, m, V, Sigma):
        m = m.T
        V = V.T
        
        nphi = np.shape(V)[0]
        Sigma_ac = np.dot(V, V.T)
        Sigma_tot = Sigma_ac + Sigma

        Sigma_tot_i = np.linalg.pinv(Sigma_tot)
        Sigma_i = np.linalg.pinv(Sigma_tot -  np.dot(np.dot(Sigma_ac, Sigma_tot_i), Sigma_ac))
        Q = Sigma_tot_i - Sigma_i
        P = np.dot(np.dot(Sigma_tot_i, Sigma_ac), Sigma_i)
        U, S, V = np.linalg.svd(P)

        Lambda = np.diag(S[np.arange(nphi)])
        Uk = U[:, np.arange(nphi)  ]
        Q_hat = np.dot(np.dot(Uk.T, Q), Uk)

        verify_iv = np.dot(Uk.T, self._verify_data.T-m[:, None])
        test_iv = np.dot(Uk.T, self._test_data.T-m[:, None])

        score_h1 = np.diag(np.dot(np.dot(verify_iv.T, Q_hat), verify_iv))
        score_h2 = np.diag(np.dot(np.dot(test_iv.T, Q_hat), test_iv))
        score_h1h2 = 2 * np.dot(np.dot(verify_iv.T, Lambda), test_iv)
        
        scores = np.add(score_h1h2, score_h1[:, None])
        scores = np.add(scores, score_h2.T)
        return scores

    def Cosine_score(self, verify_z, test_z):
        verify_z_norm = np.linalg.norm(verify_z, ord=2, axis=1)
        verify_z_data = (verify_z / verify_z_norm[:, None])

        test_z_norm = np.linalg.norm(test_z, ord=2, axis=1)
        test_z_data = (test_z / test_z_norm[:, None])

        scores = np.dot(verify_z_data , test_z_data.T)
        return scores

    def neg_Cosine_score(self, verify_z, test_z):
        verify_z_sq = np.sum(verify_z * verify_z, axis=1)
        test_z_sq = np.sum(test_z * test_z, axis=1)
        z_dot = np.dot(verify_z, test_z.T)
        scores = -0.5 * verify_z_sq[:, None] + z_dot  - 0.5 * test_z_sq[:, None].T
        return scores    
