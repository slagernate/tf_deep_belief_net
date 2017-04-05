import tensorflow as tf

#helper functions
def bernoulli_sample(x):
    """
    return tensor with element yi turned "on"
    with probability xi
    """
    return tf.ceil(x - tf.random_uniform(tf.shape(x), minval = 0, maxval=1))
