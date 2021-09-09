import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from unityagents import UnityEnvironment

from src.agent import MADDPG
from src.per import Annealing

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def evaluate(env, brain_name, agent):
    ep_rewards = []
    num_drops = 0
    while num_drops != 3:
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations
        rewards = []
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            reward = env_info.rewards
            done = env_info.local_done
            state = env_info.vector_observations
            rewards.append(reward)
            
            if any(done):
                break
        if len(rewards) != 1001:
            num_drops += 1
        rewards = np.array(rewards)
        ep_rewards.append(max(sum(rewards[:, 0]), sum(rewards[:, 1])))
    print(f'Achieved reward of {np.array(ep_rewards).mean()}!')


def train(env: UnityEnvironment, brain_name: str, num_eps: int, agent: MADDPG):
    eps = 0
    goal = 0.5
    achieved_goal = False
    previous_best = 0
    best_epoch = 0
    hist_rewards = []
    while eps < num_eps:
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        ep_rewards = []
        agent.reset_noise()
        while True:
            actions = agent.act(state)
            env_info = env.step(actions)[brain_name]
            rewards = env_info.rewards
            next_state = env_info.vector_observations
            dones = env_info.local_done
            agent.step(state, actions, rewards, next_state, dones)
            state = next_state
            ep_rewards.append(rewards)
            if np.any(dones):
                break
        eps += 1
        ep_rewards = np.array(ep_rewards)
        hist_rewards.append(max(sum(ep_rewards[:, 0]), sum(ep_rewards[:, 1])))
        # hist_rewards.append(sum(ep_rewards))
        moving_avg = np.mean(np.array(hist_rewards)[-100:])
        print(f'\rEpisode {eps}/{num_eps} \tMoving Avg: {moving_avg}\tBest Result: '
              f'{previous_best:.3f} at {best_epoch}',
              end='')
        if moving_avg > goal and not achieved_goal:
            print(f'\nAchieved {goal:.1f} after {eps - 100} episodes!')
            achieved_goal = True
            previous_best = moving_avg
        if achieved_goal and moving_avg > previous_best:
            best_epoch = eps - 100
            agent.save('../ckpt', f'ckpt_best')
        previous_best = max(previous_best, moving_avg)

    return np.array(hist_rewards)


def plot_scores(scores) -> None:
    """
    Plot scores from trained agent.
    :param scores: scores obtained after training
    """
    fig = plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('../resources/scores-plot.png')
    plt.show()


def main(gamma, a_lr, c_lr, seed, buffer_size, batch_size, num_eps, alpha, eval, tau, beta_init,
         update_every):
    env = UnityEnvironment(file_name="../world/Tennis_Linux/Tennis.x86", no_graphics=False)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
    
    beta = Annealing(beta_init, 1, 1e5)
    
    agents = MADDPG(num_agents=2, action_size=action_size, state_size=state_size, alpha=alpha,
                    update_every=update_every, beta=beta,
                    optimizers_cls=[{'actor': Adam, 'critic': Adam},
                                    {'actor': Adam, 'critic': Adam}],
                    optimizers_kwargs=[{'actor' : {'lr': a_lr[0]},
                                        'critic': {'lr': c_lr[0], 'weight_decay': 0}},
                                       {'actor' : {'lr': a_lr[1]},
                                        'critic': {'lr': c_lr[1], 'weight_decay': 0}}],
                    buffer_size=buffer_size, batch_size=batch_size, device=device, gamma=gamma,
                    soft_update_step=tau, seed=seed)
    if not eval:
        scores = train(env, brain_name, num_eps, agents)
        plot_scores(scores)
    else:
        agents.load('../ckpt', 'ckpt_1.0')
        evaluate(env, brain_name, agents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_lr", nargs='+', type=float, help='Actors learning rate')
    parser.add_argument("--c_lr", nargs='+', type=float, help='Critic learning rate')
    parser.add_argument("--gamma", default=0.99, type=float, help='Discount rate')
    parser.add_argument("--seed", default=55321, type=int, help='Random seed')
    parser.add_argument("--buffer_size", default=1e5, type=float, help='Experience Replay buffer '
                                                                       'size')
    parser.add_argument("--batch_size", default=32, type=int, help='Minibatch sample size')
    parser.add_argument("--num_eps", default=2000, type=int, help='Maximum number of episodes')
    parser.add_argument("--tau", default=1e-3, type=float, help='Soft update step-size')
    parser.add_argument("--update_every", default=1, type=int, help='Update models every ... '
                                                                    'time steps')
    parser.add_argument("--eval", const=True, nargs='?', help='Evaluate a trained model')
    
    args = parser.parse_args()
    
    main(**vars(args))
