# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import numpy as np
import tensorflow as tf

from Config import Config


class NetworkVP:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON


        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)

        #self.graph_rise = tf.Graph()
        #with self.graph_rise.as_default(), tf.device("gpu:0"):
            #self._create_graph_scope()
            #self.sess_rise = tf.Session(
                #graph=self.graph_rise,
                #config=tf.ConfigProto(
                    #allow_soft_placement=True,
                    #log_device_placement=False,
                    #gpu_options=tf.GPUOptions(allow_growth=True)))
            ##with self.sess_rise.as_default():
            #self.sess_rise.run(tf.global_variables_initializer())
            #vars = tf.global_variables()
            #self.saver_rise = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
            #self.saver_rise.restore(self.sess_rise, 'checkpoints/risetotop/network_00003000')



    def _create_graph_scope(self):
        #with tf.name_scope("risetotop"):
            #self._create_graph()
            #self.saver.restore(self.sess, "checkpoints/risetotop/network_00002001")
        self.x_rise = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.y_r_rise = tf.placeholder(tf.float32, [None], name='Yr')
        self.var_beta_rise = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate_rise = tf.placeholder(tf.float32, name='lr', shape=[])
        self.global_step_rise = tf.Variable(0, trainable=False, name='step')
        # As implemented in A3C paper
        self.n1_rise = self.conv2d_layer(self.x_rise, 8, 16, 'conv11', strides=[1, 4, 4, 1])
        self.n2_rise = self.conv2d_layer(self.n1_rise, 4, 32, 'conv12', strides=[1, 2, 2, 1])
        self.action_index_rise = tf.placeholder(tf.float32, [None, self.num_actions])
        _input = self.n2_rise
        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]
        self.flat_rise = tf.reshape(_input, shape=[-1, nb_elements._value])
        self.d1_rise = self.dense_layer(self.flat_rise, 256, 'dense1')
        self.logits_v_rise = tf.squeeze(self.dense_layer(self.d1_rise, 1, 'logits_v', func=None), axis=[1])
        self.cost_v_rise = 0.5 * tf.reduce_sum(tf.square(self.y_r_rise - self.logits_v_rise), axis=0)
        self.logits_p_rise = self.dense_layer(self.d1_rise, self.num_actions, 'logits_p', func=None)
        if Config.USE_LOG_SOFTMAX:
            self.softmax_p_rise = tf.nn.softmax(self.logits_p_rise)
            self.log_softmax_p_rise = tf.nn.log_softmax(self.logits_p_rise)
            self.log_selected_action_prob_rise = tf.reduce_sum(self.log_softmax_p_rise * self.action_index_rise, axis=1)
            self.cost_p_1_rise = self.log_selected_action_prob_rise * (self.y_r_rise - tf.stop_gradient(self.logits_v_rise))
            self.cost_p_2_rise = -1 * self.var_beta_rise * \
                        tf.reduce_sum(self.log_softmax_p_rise * self.softmax_p_rise, axis=1)
        else:
            self.softmax_p_rise = (tf.nn.softmax(self.logits_p_rise) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
            self.selected_action_prob_rise = tf.reduce_sum(self.softmax_p_rise * self.action_index_rise, axis=1)
            self.cost_p_1_rise = tf.log(tf.maximum(self.selected_action_prob_rise, self.log_epsilon)) \
                        * (self.y_r_rise - tf.stop_gradient(self.logits_v_rise))
            self.cost_p_2_rise = -1 * self.var_beta_rise * \
                        tf.reduce_sum(tf.log(tf.maximum(self.softmax_p_rise, self.log_epsilon)) *
                                      self.softmax_p_rise, axis=1)
        self.cost_p_1_agg_rise = tf.reduce_sum(self.cost_p_1_rise, axis=0)
        self.cost_p_2_agg_rise = tf.reduce_sum(self.cost_p_2_rise, axis=0)
        self.cost_p_rise = -(self.cost_p_1_agg_rise + self.cost_p_2_agg_rise)
        if Config.DUAL_RMSPROP:
            self.opt_p_rise = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate_rise,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
            self.opt_v_rise = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate_rise,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
        else:
            self.cost_all_rise = self.cost_p_rise + self.cost_v_rise
            self.opt_rise = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate_rise,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
        if Config.USE_GRAD_CLIP:
            if Config.DUAL_RMSPROP:
                self.opt_grad_v_rise = self.opt_v_rise.compute_gradients(self.cost_v_rise)
                self.opt_grad_v_clipped_rise = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v) 
                                            for g,v in self.opt_grad_v_rise if not g is None]
                self.train_op_v_rise = self.opt_v_rise.apply_gradients(self.opt_grad_v_clipped_rise)
                self.opt_grad_p_rise = self.opt_p_rise.compute_gradients(self.cost_p_rise)
                self.opt_grad_p_clipped_rise = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v)
                                            for g,v in self.opt_grad_p_rise if not g is None]
                self.train_op_p_rise = self.opt_p_rise.apply_gradients(self.opt_grad_p_clipped_rise)
                self.train_op_rise = [self.train_op_p_rise, self.train_op_v_rise]
            else:
                self.opt_grad_rise = self.opt_rise.compute_gradients(self.cost_all_rise)
                self.opt_grad_clipped_rise = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad_rise]
                self.train_op_rise = self.opt_rise.apply_gradients(self.opt_grad_clipped_rise)
        else:
            if Config.DUAL_RMSPROP:
                self.train_op_v_rise = self.opt_p_rise.minimize(self.cost_v_rise, global_step=self.global_step_rise)
                self.train_op_p_rise = self.opt_v_rise.minimize(self.cost_p_rise, global_step=self.global_step_rise)
                self.train_op_rise = [self.train_op_p_rise, self.train_op_v_rise]
            else:
                self.train_op_rise = self.opt_rise.minimize(self.cost_all_rise, global_step=self.global_step_rise)


    def _create_graph(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        # As implemented in A3C paper
        self.n1 = self.conv2d_layer(self.x, 8, 16, 'conv11', strides=[1, 4, 4, 1])
        self.n2 = self.conv2d_layer(self.n1, 4, 32, 'conv12', strides=[1, 2, 2, 1])
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])
        _input = self.n2

        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]

        self.flat = tf.reshape(_input, shape=[-1, nb_elements._value])
        self.d1 = self.dense_layer(self.flat, 256, 'dense1')

        self.logits_v = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v', func=None), axis=[1])
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

        self.logits_p = self.dense_layer(self.d1, self.num_actions, 'logits_p', func=None)
        if Config.USE_LOG_SOFTMAX:
            self.softmax_p = tf.nn.softmax(self.logits_p)
            self.log_softmax_p = tf.nn.log_softmax(self.logits_p)
            self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * self.action_index, axis=1)

            self.cost_p_1 = self.log_selected_action_prob * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(self.log_softmax_p * self.softmax_p, axis=1)
        else:
            self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
            self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1)

            self.cost_p_1 = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                        * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                                      self.softmax_p, axis=1)
        
        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)
        
        if Config.DUAL_RMSPROP:
            self.opt_p = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

            self.opt_v = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
        else:
            self.cost_all = self.cost_p + self.cost_v
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

        if Config.USE_GRAD_CLIP:
            if Config.DUAL_RMSPROP:
                self.opt_grad_v = self.opt_v.compute_gradients(self.cost_v)
                self.opt_grad_v_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v) 
                                            for g,v in self.opt_grad_v if not g is None]
                self.train_op_v = self.opt_v.apply_gradients(self.opt_grad_v_clipped)
            
                self.opt_grad_p = self.opt_p.compute_gradients(self.cost_p)
                self.opt_grad_p_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v)
                                            for g,v in self.opt_grad_p if not g is None]
                self.train_op_p = self.opt_p.apply_gradients(self.opt_grad_p_clipped)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.opt_grad = self.opt.compute_gradients(self.cost_all)
                self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad]
                self.train_op = self.opt.apply_gradients(self.opt_grad_clipped)
        else:
            if Config.DUAL_RMSPROP:
                self.train_op_v = self.opt_p.minimize(self.cost_v, global_step=self.global_step)
                self.train_op_p = self.opt_v.minimize(self.cost_p, global_step=self.global_step)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)


    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_n1", self.n1))
        summaries.append(tf.summary.histogram("activation_n2", self.n2))
        summaries.append(tf.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: x})
        return prediction

    def predict_p(self, x):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x})
        return prediction
    
    def predict_p_and_v(self, x):
        #print(self.sess_rise.run([self.softmax_p_rise, self.logits_v_rise], feed_dict={self.x_rise: x}))
        #print("shape:", x[0].shape())
        #exit()
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x})
    
    def train(self, x, y_r, a, trainer_id):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, x, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return Config.SAVE_DIRECTORY + '/%s_%08d' % (self.model_name, episode)
        #return  'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[-1])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
    
    #def get_prediction_rise(self, x):
        #print("en prediction rise")
        #states = np.zeros(
            #(Config.PREDICTION_BATCH_SIZE, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES),
            #dtype=np.float32)
        #print("state shapio:", states.shape)
        #print("shape x:", x[0].shape)
        #print("state shapio[0]:", states[0].shape)
        ##exit()
        #states[0] = x[0]
        #states[1] = x[0]
        #states[2] = x[0]
        #states[3] = x[0]
        #batch = states[:4]
        #print("batch size:", batch.shape)
        #return self.sess_rise.run([self.softmax_p_rise, self.logits_v_rise], feed_dict={self.x_rise: batch})

