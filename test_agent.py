import gym
import numpy as np
from dqn_agent import DQNAgent

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen (gunakan model terlatih jika tersedia)
agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

for e in range(5):
    state, _ = env.reset()  # Menangkap 2 nilai dari reset()
    state = np.reshape(state, [1, state_size])  # Pastikan bentuknya sesuai
    for time in range(500):
        env.render()
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)  # Menyesuaikan jumlah return
        done = done or truncated  # Gabungkan `done` dan `truncated`
        state = np.reshape(next_state, [1, state_size])
        if done:
            print(f"Test Episode: {e+1}, Score: {time}")
            break
env.close()
