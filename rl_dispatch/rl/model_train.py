import torch
import matplotlib.pyplot as plt
import os
from rl_dispatch.data_handle.data_handle import load_data_file
from rl_dispatch.environment.environment import DisplibEnvV1
from rl_dispatch.agent.agent import DoubleDQNAgent
import time

def train_double_dqn_agent(data, max_episodes:int=500, batch_size:int = 32, save_freq = 5, checkpoint_path=None, training=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    env = DisplibEnvV1(data, device, verbose=False)
    agent = DoubleDQNAgent(env, device, checkpoint_path, training)
    avg_losses = []
    avg_rewards = []
    save_episodes = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        discounted_reward = 0
        action_counts = {i: 0 for i in range(env.action_space.n)}
        gamma = agent.gamma
        step = 0
        
        while not done:
            current_idx = state.y[1].long()
            action = agent.act(state, current_idx)
            action_counts[action] += 1
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, current_idx, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            discounted_reward += (gamma ** step) * reward
            step += 1
            agent.replay(batch_size)
            
            # Terminate if more than 1K actions
            if step > 1000:
                print("Terminating episode due to exceeding 1K actions.")
                break

        
        agent.rewards.append(total_reward)
        total_actions = sum(action_counts.values())
        action_percentages = {action: round((count / total_actions) * 100) for action, count in action_counts.items()}
        print(f'Episode {episode + 1}, Epsilon: {agent.epsilon:.2f}, Total Reward: {round(total_reward)}, Discounted Reward: {round(discounted_reward)}, Actions: {action_counts}') #, Percentages: {action_percentages}')
        
        # Save a checkpoint every 10 episodes
        if ((episode + 1) % save_freq == 0) & training:
            save_episodes.append(episode + 1)
            checkpoint_save_path = f'results/checkpoint_save.pth'
            agent.save_checkpoint(checkpoint_save_path)
            print(f'Checkpoint saved at {checkpoint_save_path}')
            
            # Calculate and save average losses and rewards
            if agent.losses and agent.rewards:
                avg_loss = sum(agent.losses) / len(agent.losses)
                avg_reward = sum(agent.rewards) / len(agent.rewards)
                with open('results/average_losses.txt', 'a') as f:
                    f.write(f"Episode {episode + 1}: {avg_loss}\n")
                with open('results/.txt', 'a') as f:
                    f.write(f"Episode {episode + 1}: {avg_reward}\n")

                avg_losses.append(avg_loss)
                avg_rewards.append(avg_reward)
                
                # Plot results
                plot_results(save_episodes[:-1], avg_losses, avg_rewards)
                
                # Clear individual losses and rewards
                agent.losses.clear()
                agent.rewards.clear()
    
    # Save the final model
    agent.save_checkpoint('results/model.pth')

def plot_results(save_episodes:list, avg_losses:list, avg_rewards:list):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(save_episodes, avg_losses, label='Average Losses')
    plt.yscale('log')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Average Training Losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(save_episodes, avg_rewards, label='Average Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Average Episode Rewards')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/results_save.png')
    plt.close()

input_path = 'data/problems/toy_ex_t1_w4.json'
data = load_data_file(input_path)
checkpoint_path = 'model_saves/checkpoint_save.pth'
train_double_dqn_agent(data)