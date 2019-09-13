import numpy as np
import tensorflow as tf
def fancy_slice_2d(X, inds0, inds1):
    '''
    Tensorflow equivalent of numpy's X[inds0, inds1]
    Inputs:
    X: tf.array of size [N, k]
    inds0: tf.array of size [m,]: Indices to sample along first dimension
    inds1: tf.array of size [m,]: Indices to sample along second dimension
    Output:
    X_slice: tf.array of size [m,]
    '''
    inds0 = tf.cast(inds0, tf.int32)
    inds1 = tf.cast(inds1, tf.int32)
    shape = tf.cast(tf.shape(X), tf.int32)
    ncols = shape[1]
    X_flat = tf.reshape(X, [-1])
    X_sliced = tf.gather(X_flat, inds0 * ncols + inds1)
    return X_sliced

def normc_initializer(std=1.0, seed=7):
    '''
    Initialize array with normalized columns
    Inputs:
    std: Standard deviation for gaussians from which entry are drawn
    Output:
    _initializer: The function returns matrix has each entry gaussian with zero mean and standard deviation
                  'std' normalized to have unit norm  along axis 0 (i.e normalized columns)
    '''
    def _initializer(shape, dtype=None, partition_info=None): 
        rng_nn = np.random.RandomState(seed)
        out = rng_nn.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def sample_discrete_logits(logits):
    '''
    Use gumbell trick to sample from discrete distributions using logits
    Inputs:
    logits: tf.array of shape [N x k] corresponding to N points and k classes. We wish to sample an index
            from each row according to distribution given by tf.nn.softmax(logits,axis=1)    
    Outputs:
    indices:  tf.array of shape [N] containing sampled indices from each row    
    '''
    U = tf.random_uniform(tf.shape(logits))
    indices = tf.argmax(logits - tf.log(-tf.log(U)), axis=1)
    return indices