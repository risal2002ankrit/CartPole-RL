import gym
import gym.envs
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent= Agent(n_actions=env.action_space.n, alpha=alpha, batch_size=batch_size, 
                    n_epochs=n_epochs, input_dims=env.observation_space.shape)
    n_games = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]

    score_history = []

    learn_iters = 0
    avg_scores = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action, prob, val = agent.choose_action(observation)

            observation_, reward, done, info, *_ = env.step(action)
            n_steps += 1
            score += reward

            agent.remember(observation, action, prob, val, reward, done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            observation = observation_
        score_history.append(score)
        avg_scores = np.mean(score_history[-100:])

        if avg_scores > best_score:
            best_score = avg_scores
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_scores,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x,score_history,figure_file)





