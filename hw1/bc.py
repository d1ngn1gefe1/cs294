import argparse
import gym
import numpy as np
import os
import pickle
from sklearn.utils import shuffle
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--hidden1', type=int, default=64)
    parser.add_argument('--hidden2', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    args = parser.parse_args()

    # load data
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        expert_data = pickle.loads(f.read())
        observations = expert_data['observations']
        actions = expert_data['actions']

        # print(observations.shape)
        # print(actions.shape)

        actions = actions[:, 0]

    # build model
    input_ph = tf.placeholder('float', [None, observations.shape[1]])
    output_ph = tf.placeholder('float', [None, actions.shape[1]])

    W = {
        '1': tf.Variable(tf.random_normal([observations.shape[1], args.hidden1])),
        '2': tf.Variable(tf.random_normal([args.hidden1, args.hidden2])),
        'out': tf.Variable(tf.random_normal([args.hidden2, actions.shape[1]]))
    }
    B = {
        '1': tf.Variable(tf.random_normal([args.hidden1])),
        '2': tf.Variable(tf.random_normal([args.hidden2])),
        'out': tf.Variable(tf.random_normal([actions.shape[1]]))
    }

    x = tf.nn.relu(tf.add(tf.matmul(input_ph, W['1']), B['1']))
    x = tf.nn.relu(tf.add(tf.matmul(x, W['2']), B['2']))
    y = tf.add(tf.matmul(x, W['out']), B['out'])

    loss_op = tf.reduce_mean(tf.nn.l2_loss(y-output_ph))
    train_op = tf.train.AdamOptimizer().minimize(loss_op)

    # train
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for epoch in range(args.epoch):
        observations, actions = shuffle(observations, actions)
        loss_epoch = 0
        for step in range((observations.shape[0]-1)//args.batch_size+1):
            start = step*args.batch_size
            end = max((step+1)*args.batch_size, observations.shape[0])
            _, loss = sess.run([train_op, loss_op],
                               feed_dict={input_ph: observations[start:end], output_ph: actions[start:end]})
            loss_epoch += loss
        print('Epoch {}: loss={}'.format(epoch, loss_epoch/observations.shape[0]))

    # test
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run(y, feed_dict={input_ph: obs[None, :]})
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0:
                print('%i/%i' % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))



if __name__ == '__main__':
    main()
