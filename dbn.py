from rbm import *
#from example_rbm import *

numvis = 784 # 28x28 greyscale pixels
numhid = 784
numpen = 784
numtop = 784
numlab = 10

learning_rate = 0.0001



with tf.name_scope('dbn_layers'):
    top = tf.placeholder('float', [numtop, 1], name='dbn_top')
    pen = tf.placeholder('float', [numpen, 1], name='dbn_pen')
    hid = tf.placeholder('float', [numhid, 1], name='dbn_hid')
    vis = tf.placeholder('float', [numvis, 1], name='dbn_vis')

with tf.name_scope('dbn_weights'):
    labtop_w = tf.Variable(tf.random_normal((numtop, numlab), mean=0.0, stddev=0.01), name='dbn_labtop')
    pentop_w = tf.Variable(tf.random_normal((numtop, numpen), mean=0.0, stddev=0.01), name='dbn_pentop')
    hidpen_w = tf.Variable(tf.random_normal((numpen, numhid), mean=0.0, stddev=0.01), name='dbn_hidpen')
    vishid_w = tf.Variable(tf.zeros([numhid, numvis]), name='dbn_vishid')

with tf.name_scope('dbn_biases'):
    top_b = tf.Variable(tf.zeros([numtop, 1]), name='dbn_top_b')
    lab_b = tf.Variable(tf.zeros([numlab, 1]), name='dbn_lab_b')
    hidvis_b = tf.Variable(tf.zeros([numhid, 1]), name='dbn_hidvis_b')
    vishid_b = tf.Variable(tf.zeros([numvis, 1]), name='dbn_vishid_b')
    penhid_b = tf.Variable(tf.zeros([numpen, 1]), name='dbn_penhid_b')
    hidpen_b = tf.Variable(tf.zeros([numpen, 1]), name='dbn_hidpen_b')



# Greedily learn base layers of restricted boltzman machines 
vishid_l = rbm(numvis, numhid, learning_rate, 'vis', 'hid')
hidpen_l = rbm(numhid, numpen, learning_rate, 'hid', 'pen')
vishid_l.stack(hidpen_l)
vishid_w, hidvis_b, vishid_b = vishid_l.train_weights()
hidpen_w, penhid_b, hidpen_b = hidpen_l.train_weights()
#hidpen_w, penhid_b, hidpen_b = hidpen_l.train_weights(vishid_l)









#vishid_rbm.build_graph()
#vishid, vis_b, hid_b = vishid_rbm.train_weights()

#r1 = RBM(numvis, numhid, gibbs_k=5, learning_rate=learning_rate, verbose=1)
#r1._create_graph()
#
#
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
#r1.fit(mnist.train)


