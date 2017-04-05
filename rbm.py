
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tf_utils import *


"""
Architecture:

 --top---
 |       |
pen     labels
 |
hid
 |
vis
"""


class rbm(object):

    def __init__(self, numvis, numhid, learning_rate, actual_vis_layer_name='vis', actual_hid_layer_name='hid'):
        self.numvis = numvis
        self.numhid = numhid
        self.vis_name=actual_vis_layer_name
        self.hid_name=actual_hid_layer_name
        self.name = self.vis_name + self.hid_name 

        with tf.name_scope(self.vis_name+'_layer'):
            self.input_vec = None
            self.vis_b_target = None

        with tf.name_scope(self.hid_name+'_layer'):
            self.hid_b_target = None
            self.hid_b = None
            self.learning_rate = learning_rate 
        
        self.vishid_w = None
        self.vishid_w_target = None
        self.first_hid_node_weights = None

        self.below = None
        self.above = None

        self.saver = None

        self.graph = tf.Graph()
        self.sess = None

    def build_graph(self):
        # Train vis<->hid RBM via contrastive divergence (CD)
        CD=5

        with tf.name_scope(self.vis_name+'_layer'):
            vis_b = tf.Variable(tf.truncated_normal((self.numvis, 1), mean=0.0, stddev=0.001), name=self.vis_name+'_b')
            self.input_vec = tf.placeholder('float', [1, 784], name=self.vis_name+'_input_vec')
            bin_input_vec = tf.ceil(self.input_vec - 0.5)
            lower = self.below
            if (not lower):
                lower_layer_output = tf.reshape(bin_input_vec, [784, 1])
            else:
                lower_layer_output = lower.sample_layer(bin_input_vec)
                        
            tmp_vis = lower_layer_output
            visprobs = None
            #tmp_vis = tf.transpose(lower_layer_output)

        with tf.name_scope(self.hid_name+'_layer'):
            self.hid_b = tf.Variable(tf.truncated_normal((self.numhid, 1), mean=0.0, stddev=0.001), name=self.hid_name+'_b')
            hidprobs0 = None
            hidprobs = None
            hid = None

        with tf.name_scope(self.name+'_weights'):
            self.vishid_w = tf.Variable(tf.truncated_normal((self.numhid, self.numvis), mean=0.0, stddev=0.001), name=self.name+'_w')
            self.vishid_w_target = self.vishid_w
         
        for step in range(CD):
            with tf.name_scope(self.hid_name+'_layer'):
                hidprobs = tf.sigmoid(tf.matmul(self.vishid_w, tmp_vis) + self.hid_b)
                hid = bernoulli_sample(hidprobs)

            if (step == 0):
                with tf.name_scope(self.hid_name+'_layer'):
                    hidprobs0 = hidprobs
                    #    pos_stats = tf.matmul(hid, tf.transpose(tmp_vis))
                    #elif (step == (CD-1)):
                    #    neg_stats = tf.matmul(hid, tf.transpose(tmp_vis))

            visprobs = tf.sigmoid(tf.matmul(tf.transpose(self.vishid_w), hid) + vis_b)
            tmp_vis = bernoulli_sample(visprobs)

        # update self.vishid_w weights
        pos_stats = tf.matmul(hidprobs0, tf.transpose(lower_layer_output))
        neg_stats = tf.matmul(hidprobs, tf.transpose(tmp_vis))
        self.vishid_w_target = tf.assign_add(self.vishid_w, self.learning_rate*(pos_stats-neg_stats))
        self.vis_b_target  = tf.assign_add(vis_b, tf.mul(self.learning_rate, tf.subtract(lower_layer_output, tmp_vis)))
        self.hid_b_target  = tf.assign_add(self.hid_b, tf.mul(self.learning_rate, tf.subtract(hidprobs0, hidprobs)))

        # visualizations
        #visprobs_image = tf.reshape(visprobs, [1, 28, 28, 1])
        #tf.summary.image(self.vis_name+"probs", visprobs_image, max_outputs=1)

        vishid_w_square = tf.reshape(self.vishid_w_target, [-1, 28, 28, 1])
        self.first_hid_node_weights = tf.slice(vishid_w_square, [0, 0, 0, 0], [10, 28, 28, 1])
        tf.summary.image(self.name+'_w', self.first_hid_node_weights, max_outputs=10)
        tf.summary.histogram(self.name + " weights", self.vishid_w_target)
        tf.summary.histogram(self.vis_name + " biases", self.vis_b_target)
        tf.summary.histogram(self.hid_name + " biases", self.hid_b_target)

        self.saver = tf.train.Saver({self.vishid_w.name: self.vishid_w, self.hid_b.name: self.hid_b})

    def train_weights(self):

        tf.reset_default_graph()
        self.build_graph()

        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # Weights and biases needed to train next layer

        self.sess = tf.InteractiveSession()

        train_writer = tf.summary.FileWriter("log/"+self.name, self.sess.graph)

        merged = tf.summary.merge_all()
        tf.global_variables_initializer().run()

        #mnist_train_batches = 55000
        batch_count = 1000
        batch_print = batch_count/10
        batch_size = 1
        completion_decade = 0
        print("training " + self.name + " rbm...")
        for batch_n in tqdm(range(batch_count/batch_size)):
            data, labels = mnist.train.next_batch(batch_size)
            #img = tf.summary.image(self.name+'_w1', self.first_hid_node_weights, max_outputs=2)
            #hist = tf.summary.histogram(self.name+'_w', self.vishid_w_target)
            updates = [merged, self.vishid_w_target, self.vis_b_target, self.hid_b_target]
            #summary, img_summary, _, _, _ = self.sess.run(updates, feed_dict={self.input_vec: data})
            #updates = [img, hist, self.vishid_w_target, self.vis_b_target, self.hid_b_target]
            #img_summary, hist_summary, _, _, _ = self.sess.run(updates, feed_dict={self.input_vec: data})
            summary, _, _, _ = self.sess.run(updates, feed_dict={self.input_vec: data})
            if (batch_n % (batch_print) == 0):
                #print("%d %% complete" % (completion_decade * 10))
                completion_decade += 1
                #train_writer.add_summary(img_summary, batch_n)
                #train_writer.add_summary(hist_summary, batch_n)
            train_writer.add_summary(summary, batch_n)

        #print("100% complete")


        save_path = self.saver.save(self.sess, "log/weights_and_biases_from_"+self.name)
        train_writer.close()
        self.sess.close()
        #if (not self.above):
        #    self.close_sess()

        return self.vishid_w_target, self.vis_b_target, self.hid_b_target

    #def close_sess():
    #    if (not self.below):
    #        self.sess.close()
    #    else:
    #        lower.close_sess()
    #        self.close_sess()
        
    def stack(self, layer):
        self.above = layer
        layer.below = self
        
    def sample_layer(self, x):
        input_x = None
        if (not self.below):
            input_x = tf.reshape(x, [784, 1])
        else:
            input_x = self.below.sample_layer(x)
        #with self.graph.as_default():
        self.sess = tf.Session()
        self.saver.restore(self.sess, "log/weights_and_biases_from_"+self.name)
            #initial_weights = tmp_session.run(self.vidhiw_w.initialized_value())
        return bernoulli_sample(tf.sigmoid(tf.matmul(self.vishid_w, input_x) + self.hid_b))
        

