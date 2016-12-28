# Inspire by https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

import numpy as np
import cPickle as pickle
import gym
import tensorflow as tf

# hyperparameters
H = 20  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-2
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # False # resume from previous checkpoint?
render = False  # True#False

# model initialization
# D = 80 * 80 # input dimensionality: 80x80 grid
D = 4  # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        # if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


# env = gym.make("Pong-v0")
env = gym.make("CartPole-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

sess = tf.InteractiveSession()
# model['W1'] = np.random.randn(h,d) / np.sqrt(d) # "xavier" initialization
# model['w2'] = np.random.randn(h) / np.sqrt(h)
# h = np.dot(model['w1'], x)
# h[h<0] = 0 # relu nonlinearity
# logp = np.dot(model['w2'], h)
# p = sigmoid(logp)

with tf.variable_scope("value"):
    state = tf.placeholder(tf.float64, [None, 4])
    newvals = tf.placeholder(tf.float64, [None, 1])
    w1 = tf.get_variable("w1", initializer=tf.constant(np.random.randn(D, H) / np.sqrt(D)), dtype=tf.float64)
    tf.histogram_summary("w1", w1)
    h1 = tf.nn.relu(tf.matmul(state, w1))
    w2 = tf.get_variable("w2", initializer=tf.constant(np.random.randn(H, 1) / np.sqrt(H)), dtype=tf.float64)
    tf.histogram_summary("w2", w2)
    calculated = tf.nn.sigmoid(tf.matmul(h1, w2))
    # diffs = calculated - newvals
    # loss = tf.nn.l2_loss(diffs)
    loss = -tf.reduce_sum(newvals * tf.log(calculated))
    tf.scalar_summary('loss', loss)
    # loss = -tf.reduce_sum(newvals*tf.log(calculated) + (1-newvals)*tf.log(1-calculated))
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)
    # return calculated, state, newvals, optimizer, loss

summaries_dir = '/tmp/tf_logs'
if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)
merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(summaries_dir, sess.graph)
sess.run(tf.initialize_all_variables())
while True:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    # cur_x = prepro(observation)
    # x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    # prev_x = cur_x
    x = observation

    # forward the policy network and sample an action from the returned probability
    # aprob, h = policy_forward(x)
    x = np.array([x])
    aprob = sess.run(calculated, feed_dict={state: x})
    # action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
    action = 1 if np.random.uniform() < aprob else 0  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    # hs.append(h) # hidden state
    # y = 1 if action == 2 else 0 # a "fake label"
    y = 1 if action == 1 else 0  # a "fake label"
    dlogps.append(
        y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward = -1. if done else reward
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done or reward_sum >= 200:  # an episode finished
        episode_number += 1

        # summary_str = sess.run(merged_summary_op)
        # summary_writer.add_summary(summary_str, episode_number)
        # summary_writer.add_summary('episode_number', episode_number)
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        # eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        # xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)

        sum_str, _ = sess.run([merged_summary_op, optimizer], feed_dict={state: epx, newvals: epdlogp})
        summary_writer.add_summary(sum_str, episode_number)
        # grad = policy_backward(eph, epdlogp)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory
        # for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch
        #
        # # perform rmsprop parameter update every batch_size episodes
        # if episode_number % batch_size == 0:
        #   for k,v in model.iteritems():
        #     g = grad_buffer[k] # gradient
        #     rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        #     model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        #     grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print '%d: resetting env. episode reward total was %f. running mean: %f' % (
        episode_number, reward_sum, running_reward)
        # print 'loss=', loss, calculated
        # if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

        # if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        #   print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
