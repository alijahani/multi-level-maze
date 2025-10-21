import gymnasium as gym
import multi_level_maze

if __name__ == "__main__":
    env = gym.make("MultiLevelMaze-v0", size=3, levels=3, maze_seed=1, max_steps=1000, cell_size=30, render_fps=15)

    total_reward = 0
    obs, info = env.reset(seed=1)
    while True:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    env.close()
    print(f"Total reward: {total_reward}")
