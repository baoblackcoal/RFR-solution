# Inspire by https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
#            https://github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py
import numpy as np
import gym
import tensorflow as tf

# hyperparameters
H = 20  # number of hidden layer neurons
D = 4  # input dimensionality
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-2
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
render = False  # True#False


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        # if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


env = gym.make("CartPole-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

sess = tf.InteractiveSession()

with tf.variable_scope("policy"):
    state = tf.placeholder(tf.float64, [None, 4])
    advantage = tf.placeholder(tf.float64, [None, 1])
    w1 = tf.get_variable("w1", initializer=tf.constant(np.random.randn(D, H) / np.sqrt(D)), dtype=tf.float64)
    tf.histogram_summary("w1", w1)
    h1 = tf.nn.relu(tf.matmul(state, w1))
    w2 = tf.get_variable("w2", initializer=tf.constant(np.random.randn(H, 1) / np.sqrt(H)), dtype=tf.float64)
    tf.histogram_summary("w2", w2)
    probability = tf.nn.sigmoid(tf.matmul(h1, w2))
    loss = -tf.reduce_sum(advantage * tf.log(probability))
    tf.scalar_summary("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)

summaries_dir = "/tmp/tf_logs"
if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)
merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(summaries_dir, sess.graph)
sess.run(tf.initialize_all_variables())
for i in xrange(1000):
    if render: env.render()

    x = observation

    # forward the policy network and sample an action from the returned probability
    x = np.array([x])
    aprob = sess.run(probability, feed_dict={state: x})
    action = 1 if np.random.uniform() < aprob else 0  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    y = 1 if action == 1 else 0  # a "fake label"

    # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    dlogps.append(y - aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward = -1. if done else reward
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done or reward_sum >= 200:  # an episode finished
        episode_number += 1

        # summary_str = sess.run(merged_summary_op)
        # summary_writer.add_summary(summary_str, episode_number)
        # summary_writer.add_summary("episode_number", episode_number)
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        # eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)

        sum_str, _ = sess.run([merged_summary_op, optimizer], feed_dict={state: epx, advantage: epdlogp})
        summary_writer.add_summary(sum_str, episode_number)

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print "%d: resetting env. episode reward total was %f. running mean: %f" % (
            episode_number, reward_sum, running_reward)
        # print "loss=", loss, probability
        # if episode_number % 100 == 0: pickle.dump(model, open("save.p", "wb"))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None
