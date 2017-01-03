# https://github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py

import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt

learning_rate = 1e-3

def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters", [4, 2])
        state = tf.placeholder("float", [None, 4])
        actions = tf.placeholder("float", [None, 2])
        advantages = tf.placeholder("float", [None, 1])
        reward_input = tf.placeholder("float")
        episode_reward = tf.get_variable("episode_reward", initializer=tf.constant(0.))
        episode_reward = reward_input
        linear = tf.matmul(state, params)
        probabilities = tf.nn.softmax(linear)
        good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        tf.scalar_summary("loss", loss)
        tf.scalar_summary("episode_reward", episode_reward)
        return probabilities, state, actions, advantages, optimizer, reward_input, episode_reward


def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float", [None, 4])
        newvals = tf.placeholder("float", [None, 1])
        w1 = tf.get_variable("w1", [4, 10])
        b1 = tf.get_variable("b1", [10])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable("w2", [10, 1])
        b2 = tf.get_variable("b2", [1])
        calculated = tf.matmul(h1, w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        # tf.scalar_summary(loss.name, loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return calculated, state, newvals, optimizer, loss


summaries_dir = "/tmp/actor_critic"
sess = tf.InteractiveSession()
if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)
policy_grad = policy_gradient()
value_grad = value_gradient()
merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(summaries_dir, sess.graph)
sess.run(tf.initialize_all_variables())


def run_episode(env, policy_grad, value_grad, sess, episode_number):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer, pl_reward_input, pl_episode_reward = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    for _ in xrange(200):
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated, feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0, 1) < probs[0][0] else 1
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in xrange(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated, feed_dict={vl_state: obs_vector})[0][0]

        # advantage: how much better was this action than normal
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    advantages_vector = np.expand_dims(advantages, axis=1)
    sum_str, _, _ = sess.run([merged_summary_op, pl_optimizer, pl_episode_reward],
                             feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions,
                                        pl_reward_input: totalreward})
    summary_writer.add_summary(sum_str, episode_number)

    return totalreward


env = gym.make('CartPole-v0')
# env.monitor.start('cartpole-hill/', force=True)
reward_sum = 0
running_reward = None
episode_number = 0
while True:
    episode_number += 1
    reward = run_episode(env, policy_grad, value_grad, sess, episode_number)
    reward_sum = reward
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print '%d: resetting env. episode reward total was %f. running mean: %f' % (
        episode_number, reward_sum, running_reward)
    reward_sum = 0

# env.monitor.close()
