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

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from Config import Config
from Environment import Environment
from Experience import Experience

# hierarchical
import tensorflow as tf
from Config import Config

class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.env = Environment()
        self.num_actions = self.env.get_num_actions()
        self.actions = np.arange(self.num_actions)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

        self.graph_rise = tf.Graph()
        with self.graph_rise.as_default(), tf.device("cpu:0"):
            self._create_graph_scope()
            self.sess_rise = tf.Session(
                graph=self.graph_rise,
                config=tf.ConfigProto(
                    allow_soft_placement=False,
                    log_device_placement=False,
                    gpu_options=tf.GPUOptions(allow_growth=True)))
            #with self.sess_rise.as_default():
            self.sess_rise.run(tf.global_variables_initializer())
            vars = tf.global_variables()
            self.saver_rise = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
            self.saver_rise.restore(self.sess_rise, 'checkpoints/risetotop/network_00003000')


    def _create_graph_scope(self):
        #with tf.name_scope("risetotop"):
            #self._create_graph()
            #self.saver.restore(self.sess, "checkpoints/risetotop/network_00002001")
        self.x_rise = tf.placeholder(
            tf.float32, [None, 168, 168, 4], name='X')
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
            self.cost_p_1_rise = tf.log(tf.maximum(self.selected_action_prob_rise, 1e-6)) \
                        * (self.y_r_rise - tf.stop_gradient(self.logits_v_rise))
            self.cost_p_2_rise = -1 * self.var_beta_rise * \
                        tf.reduce_sum(tf.log(tf.maximum(self.softmax_p_rise, 1e-6)) *
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
    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:]

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    def predict(self, state):
        # put the state in the prediction q
        #print("sssstate:", state.shape)
        #exit()
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        return p, v

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        while not done:
            # very first few frames
            if self.env.current_state is None:
                self.env.step(-1)  # 0 == NOOP
                continue

            prediction, value = self.predict(self.env.current_state)
            action = self.select_action(prediction)
            reward, done = self.env.step(action)
            reward_sum += reward
            exp = Experience(self.env.previous_state, action, prediction, reward, done)
            experiences.append(exp)

            states = np.zeros(
                (Config.PREDICTION_BATCH_SIZE, Config.IMAGE_HEIGHT,
                 Config.IMAGE_WIDTH, Config.STACKED_FRAMES),
                dtype=np.float32)
            states[0] = self.env.current_state
            batch = states[:1]
            print("shape state:", states.shape)
            print("prediction:", self.sess_rise.run(self.softmax_p_rise, feed_dict={self.x_rise: batch}))

            if done or time_count == Config.TIME_MAX:
                terminal_reward = reward if done else value

                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_ = self.convert_data(updated_exps)
                yield x_, r_, a_, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_))
            self.episode_log_q.put((datetime.now(), total_reward, total_length))
