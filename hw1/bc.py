import argparse
import gym
from keras.layers import Dense
from keras.models import load_model, Sequential
import numpy as np
import os
import pickle
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    args = parser.parse_args()

    model_path = 'models/{}.h5'.format(args.envname)
    if os.path.isfile(model_path):
        model = load_model(model_path)
    else:
        # load data
        with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
            expert_data = pickle.loads(f.read())
            observations = expert_data['observations']
            actions = expert_data['actions'][:, 0]
            print(observations.shape, actions.shape)

        # train model
        model = Sequential()
        model.add(Dense(args.hidden_sizes[0], activation='relu', input_shape=(observations.shape[1],)))
        model.add(Dense(args.hidden_sizes[1], activation='relu'))
        model.add(Dense(actions.shape[1], activation='linear'))

        model.compile('adam', loss='mean_squared_error')
        model.fit(observations, actions, batch_size=args.batch_size, epochs=args.epochs)
        model.save(model_path)

    # test model
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    for i in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.predict(obs[None, :])
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
        print('iter {}: done at step {}/{}'.format(i, steps, max_steps))
        returns.append(totalr)

    log_path = 'results/{}.txt'.format(args.envname)
    log = 'returns: {}\nmean return: {}\nstd of return: {}'.format(returns, np.mean(returns), np.std(returns))
    print(log)
    log_file = open(log_path, 'w')
    log_file.write(log)
    log_file.close()


if __name__ == '__main__':
    main()
